"""Tests for the matched synthetic model-versus-static harness."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from agent.config import load_config
from agent.nodes import AssessmentResult
from agent.state import (
    DecisionAction,
    DecisionSource,
    PhasingOutcome,
)
from evaluation.matched_comparison import (
    MECHANICS_RETRY_STRATEGY,
    MatchedComparisonBatch,
    MatchedComparisonError,
    run_matched_synthetic_comparison,
    write_matched_comparison,
)


class DeterministicModelFixture:
    """Model-shaped test double following the declared fake case labels."""

    def assess(self, state):
        if state.evidence.outcome == PhasingOutcome.PHASED:
            action = DecisionAction.ACCEPT
            strategy = None
        elif (
            state.evidence.outcome == PhasingOutcome.PARTIALLY_PHASED
            and MECHANICS_RETRY_STRATEGY.strategy_id not in state.strategies_tried
        ):
            action = DecisionAction.RETRY
            strategy = MECHANICS_RETRY_STRATEGY
        else:
            action = DecisionAction.ESCALATE
            strategy = None
        return AssessmentResult(
            action=action,
            confidence=1.0,
            rationale="Synthetic model test double; no biological meaning.",
            source=DecisionSource.LLM,
            retry_strategy=strategy,
        )


class MatchedComparisonTests(TestCase):
    def test_requires_repeated_trials(self) -> None:
        with self.assertRaises(MatchedComparisonError):
            run_matched_synthetic_comparison(
                config=load_config(),
                repetitions=1,
                model_assessor=DeterministicModelFixture(),
            )

    def test_both_systems_run_identical_cases_and_repetitions(self) -> None:
        config = load_config()
        original_strategies = list(config.pipeline.retry_strategies)

        batch = run_matched_synthetic_comparison(
            config=config,
            repetitions=2,
            model_assessor=DeterministicModelFixture(),
        )

        self.assertEqual(config.pipeline.retry_strategies, original_strategies)
        self.assertEqual(batch.data_status, "synthetic")
        self.assertFalse(batch.scientific_interpretation_allowed)
        self.assertFalse(batch.evaluation.scientifically_valid)
        self.assertEqual(len(batch.cases), 3)
        self.assertEqual(len(batch.predictions), 12)
        self.assertEqual(len(batch.run_records), 12)
        self.assertTrue(
            all("scenario" not in record.initial_state.metadata for record in batch.run_records)
        )
        slots_by_system = {}
        for prediction in batch.predictions:
            slots_by_system.setdefault(prediction.system_name, set()).add(
                (prediction.case_id, prediction.repetition)
            )
        self.assertEqual(len(slots_by_system), 2)
        self.assertEqual(*slots_by_system.values())

        states_by_scenario = {}
        for record in batch.run_records:
            states_by_scenario.setdefault(record.scenario, set()).add(
                record.initial_state.model_dump_json(exclude_computed_fields=True)
            )
        self.assertTrue(
            all(len(states) == 1 for states in states_by_scenario.values())
        )
        for metrics in batch.evaluation.systems.values():
            self.assertEqual(metrics.prediction_count, 6)
            self.assertEqual(metrics.case_coverage, 1.0)
            self.assertEqual(metrics.initial_action_accuracy, 1.0)

        retry_records = [
            record
            for record in batch.run_records
            if record.scenario == "retry-success"
        ]
        self.assertEqual(len(retry_records), 4)
        self.assertTrue(
            all(record.final_state.retries_used == 1 for record in retry_records)
        )
        self.assertTrue(
            all(
                record.final_state.decision_history[0].action
                == DecisionAction.RETRY
                for record in retry_records
            )
        )

    def test_shared_report_round_trips_through_schema(self) -> None:
        batch = run_matched_synthetic_comparison(
            config=load_config(),
            repetitions=2,
            model_assessor=DeterministicModelFixture(),
        )

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "comparison.json"
            write_matched_comparison(batch, path)
            reloaded = MatchedComparisonBatch.model_validate_json(path.read_text())

        self.assertEqual(reloaded.batch_id, batch.batch_id)
        self.assertEqual(
            reloaded.retry_strategy.parameters["scope"],
            "mechanics-test-only",
        )

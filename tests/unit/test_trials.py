"""Unit tests for repeated model consistency trial orchestration."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from agent.config import load_config
from agent.run import run_synthetic_case
from evaluation.trials import (
    ConsistencyTrialBatch,
    TrialBatchError,
    run_model_consistency_trials,
    write_trial_batch,
)


class ConsistencyTrialTests(TestCase):
    def test_at_least_two_repetitions_are_required(self) -> None:
        with self.assertRaises(TrialBatchError):
            run_model_consistency_trials(
                config=load_config(),
                scenario="escalate",
                repetitions=1,
            )

    def test_successful_repetitions_are_normalized_and_scored(self) -> None:
        config = load_config()
        synthetic_record = run_synthetic_case(
            config=config,
            scenario="escalate",
            assessor_kind="scripted",
        )

        with patch(
            "evaluation.trials.run_synthetic_case",
            return_value=synthetic_record,
        ) as runner:
            batch = run_model_consistency_trials(
                config=config,
                scenario="escalate",
                repetitions=3,
            )

        self.assertEqual(len(batch.predictions), 3)
        self.assertEqual(len(batch.run_records), 3)
        self.assertEqual(batch.decision_distribution, {"escalate": 3})
        metrics = next(iter(batch.evaluation.systems.values()))
        self.assertEqual(metrics.mean_repeat_consistency, 1.0)
        fingerprints = {
            call.kwargs["initial_state"].model_dump_json(
                exclude_computed_fields=True
            )
            for call in runner.call_args_list
        }
        self.assertEqual(len(fingerprints), 1)

    def test_model_errors_are_recorded_instead_of_aborting_batch(self) -> None:
        with patch(
            "evaluation.trials.run_synthetic_case",
            side_effect=RuntimeError("synthetic provider failure"),
        ):
            batch = run_model_consistency_trials(
                config=load_config(),
                scenario="escalate",
                repetitions=2,
            )

        self.assertEqual(batch.decision_distribution, {"error": 2})
        self.assertEqual(len(batch.run_records), 0)
        self.assertTrue(all(item.error_message for item in batch.predictions))
        metrics = next(iter(batch.evaluation.systems.values()))
        self.assertEqual(metrics.error_rate, 1.0)

    def test_written_batch_round_trips_through_schema(self) -> None:
        config = load_config()
        synthetic_record = run_synthetic_case(
            config=config,
            scenario="escalate",
            assessor_kind="scripted",
        )
        with patch(
            "evaluation.trials.run_synthetic_case",
            return_value=synthetic_record,
        ):
            batch = run_model_consistency_trials(
                config=config,
                scenario="escalate",
                repetitions=2,
            )

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "batch.json"
            write_trial_batch(batch, path)
            reloaded = ConsistencyTrialBatch.model_validate_json(path.read_text())

        self.assertEqual(reloaded.batch_id, batch.batch_id)

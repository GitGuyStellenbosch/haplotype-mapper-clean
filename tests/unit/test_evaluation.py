"""Unit tests for evaluation records and classification metrics."""

from unittest import TestCase

from agent.config import load_config
from agent.run import run_synthetic_case
from agent.state import DecisionAction
from evaluation.metrics import (
    Difficulty,
    EvaluationCase,
    EvaluationError,
    LabelProvenance,
    PredictionRecord,
    evaluate_predictions,
    prediction_from_run_record,
)


def cases() -> list[EvaluationCase]:
    return [
        EvaluationCase(
            case_id="easy-accept",
            expected_initial_action=DecisionAction.ACCEPT,
            expected_final_action="accept",
            difficulty=Difficulty.EASY,
            label_provenance=LabelProvenance.SYNTHETIC,
        ),
        EvaluationCase(
            case_id="hard-retry",
            expected_initial_action=DecisionAction.RETRY,
            expected_final_action="accept",
            difficulty=Difficulty.HARD,
            label_provenance=LabelProvenance.SYNTHETIC,
        ),
        EvaluationCase(
            case_id="hard-escalate",
            expected_initial_action=DecisionAction.ESCALATE,
            expected_final_action="escalate",
            difficulty=Difficulty.HARD,
            label_provenance=LabelProvenance.SYNTHETIC,
        ),
    ]


def prediction(
    case_id: str,
    system_name: str,
    initial_action: DecisionAction,
    final_action: str,
    *,
    repetition: int = 1,
) -> PredictionRecord:
    return PredictionRecord(
        case_id=case_id,
        system_name=system_name,
        repetition=repetition,
        run_id=f"{system_name}-{case_id}-{repetition}",
        predicted_initial_action=initial_action,
        predicted_final_action=final_action,
        retries_used=1 if initial_action == DecisionAction.RETRY else 0,
    )


class EvaluationTests(TestCase):
    def test_two_systems_are_scored_on_the_same_cases(self) -> None:
        predictions = [
            prediction("easy-accept", "static", DecisionAction.ACCEPT, "accept"),
            prediction("hard-retry", "static", DecisionAction.ESCALATE, "escalate"),
            prediction("hard-escalate", "static", DecisionAction.ESCALATE, "escalate"),
            prediction("easy-accept", "agent", DecisionAction.ACCEPT, "accept"),
            prediction("hard-retry", "agent", DecisionAction.RETRY, "accept"),
            prediction("hard-escalate", "agent", DecisionAction.ESCALATE, "escalate"),
        ]

        report = evaluate_predictions(cases(), predictions)

        self.assertFalse(report.scientifically_valid)
        self.assertEqual(report.systems["agent"].initial_action_accuracy, 1.0)
        self.assertEqual(report.systems["agent"].macro_f1, 1.0)
        self.assertAlmostEqual(
            report.systems["static"].initial_action_accuracy, 2 / 3
        )
        self.assertEqual(
            report.systems["static"].accuracy_by_difficulty["hard"], 0.5
        )
        self.assertEqual(
            report.systems["static"].action_metrics["escalate"].precision,
            0.5,
        )

    def test_repeat_consistency_detects_variable_output(self) -> None:
        single_case = [cases()[0]]
        predictions = [
            prediction(
                "easy-accept", "variable", DecisionAction.ACCEPT, "accept"
            ),
            prediction(
                "easy-accept",
                "variable",
                DecisionAction.ESCALATE,
                "escalate",
                repetition=2,
            ),
        ]

        report = evaluate_predictions(single_case, predictions)

        self.assertEqual(
            report.systems["variable"].mean_repeat_consistency,
            0.5,
        )

    def test_latency_reports_mean_and_median(self) -> None:
        predictions = [
            prediction(
                "easy-accept",
                "timed",
                DecisionAction.ACCEPT,
                "accept",
                repetition=index,
            ).model_copy(update={"duration_seconds": duration})
            for index, duration in enumerate((90.0, 10.0, 11.0), start=1)
        ]

        metrics = evaluate_predictions([cases()[0]], predictions).systems["timed"]

        self.assertEqual(metrics.mean_duration_seconds, 37.0)
        self.assertEqual(metrics.median_duration_seconds, 11.0)

    def test_duplicate_prediction_is_rejected(self) -> None:
        duplicate = prediction(
            "easy-accept", "agent", DecisionAction.ACCEPT, "accept"
        )

        with self.assertRaises(EvaluationError):
            evaluate_predictions(cases(), [duplicate, duplicate])

    def test_unknown_case_prediction_is_rejected(self) -> None:
        unknown = prediction(
            "not-a-case", "agent", DecisionAction.ACCEPT, "accept"
        )

        with self.assertRaises(EvaluationError):
            evaluate_predictions(cases(), [unknown])

    def test_non_synthetic_label_requires_provenance_notes(self) -> None:
        with self.assertRaises(ValueError):
            EvaluationCase(
                case_id="real-case",
                expected_initial_action=DecisionAction.ESCALATE,
                difficulty=Difficulty.HARD,
                label_provenance=LabelProvenance.DERIVED_MAPPING,
            )

    def test_run_record_preserves_initial_retry_and_final_acceptance(self) -> None:
        run_record = run_synthetic_case(
            config=load_config(),
            scenario="retry-success",
            assessor_kind="scripted",
        )

        normalized = prediction_from_run_record(
            run_record,
            case_id="hard-retry",
            system_name="agent",
            repetition=1,
        )

        self.assertEqual(normalized.predicted_initial_action, DecisionAction.RETRY)
        self.assertEqual(normalized.predicted_final_action, "accept")
        self.assertEqual(normalized.retries_used, 1)

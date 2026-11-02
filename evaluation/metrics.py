"""Validated evaluation records and dependency-free classification metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from statistics import mean, median
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.run import RunRecord
from agent.state import DecisionAction


class EvaluationError(ValueError):
    """Raised when evaluation inputs are incomplete or internally inconsistent."""


class Difficulty(str, Enum):
    """Predeclared case subset used by the research hypothesis."""

    EASY = "easy"
    HARD = "hard"


class LabelProvenance(str, Enum):
    """Origin of an expected decision label."""

    SYNTHETIC = "synthetic"
    EXPERT_ANNOTATED = "expert_annotated"
    DERIVED_MAPPING = "derived_mapping"


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EvaluationCase(EvaluationModel):
    """One labelled decision problem shared by every compared system."""

    case_id: str = Field(min_length=1)
    expected_initial_action: DecisionAction
    expected_final_action: Literal["accept", "escalate"] | None = None
    difficulty: Difficulty
    label_provenance: LabelProvenance
    label_notes: str = ""

    @model_validator(mode="after")
    def non_synthetic_labels_require_notes(self) -> "EvaluationCase":
        if self.label_provenance != LabelProvenance.SYNTHETIC and not self.label_notes:
            raise ValueError("non-synthetic labels require provenance notes")
        return self


class PredictionRecord(EvaluationModel):
    """Normalized result of one system repetition on one case."""

    case_id: str = Field(min_length=1)
    system_name: str = Field(min_length=1)
    repetition: int = Field(ge=1)
    run_id: str = Field(min_length=1)
    predicted_initial_action: DecisionAction | None
    predicted_final_action: Literal["accept", "escalate"] | None
    retries_used: int = Field(ge=0)
    duration_seconds: float | None = Field(default=None, ge=0.0)
    error_message: str | None = None

    @model_validator(mode="after")
    def errors_and_predictions_are_consistent(self) -> "PredictionRecord":
        has_actions = (
            self.predicted_initial_action is not None
            or self.predicted_final_action is not None
        )
        if self.error_message is not None and has_actions:
            raise ValueError("failed predictions cannot also contain decision actions")
        if self.error_message is None and not has_actions:
            raise ValueError("successful predictions must contain decision actions")
        return self


class ActionMetrics(EvaluationModel):
    precision: float
    recall: float
    f1: float
    support: int


class SystemMetrics(EvaluationModel):
    system_name: str
    prediction_count: int
    case_coverage: float
    error_rate: float
    initial_action_accuracy: float
    final_action_accuracy: float | None
    macro_f1: float
    action_metrics: dict[str, ActionMetrics]
    accuracy_by_difficulty: dict[str, float]
    mean_retries: float
    mean_duration_seconds: float | None
    median_duration_seconds: float | None = None
    mean_repeat_consistency: float


class EvaluationReport(EvaluationModel):
    generated_at: datetime
    case_count: int
    scientifically_valid: bool
    label_provenance_counts: dict[str, int]
    systems: dict[str, SystemMetrics]


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _action_metrics(
    cases_by_id: dict[str, EvaluationCase],
    predictions: list[PredictionRecord],
) -> dict[str, ActionMetrics]:
    metrics: dict[str, ActionMetrics] = {}
    for action in DecisionAction:
        true_positive = 0
        false_positive = 0
        false_negative = 0
        support = 0
        for prediction in predictions:
            expected = cases_by_id[prediction.case_id].expected_initial_action
            predicted = prediction.predicted_initial_action
            if expected == action:
                support += 1
            if expected == action and predicted == action:
                true_positive += 1
            elif expected != action and predicted == action:
                false_positive += 1
            elif expected == action and predicted != action:
                false_negative += 1

        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        metrics[action.value] = ActionMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
        )
    return metrics


def _repeat_consistency(predictions: list[PredictionRecord]) -> float:
    grouped: dict[str, list[str]] = defaultdict(list)
    for prediction in predictions:
        value = (
            prediction.predicted_initial_action.value
            if prediction.predicted_initial_action is not None
            else "error"
        )
        grouped[prediction.case_id].append(value)
    scores = [
        max(Counter(values).values()) / len(values)
        for values in grouped.values()
    ]
    return mean(scores) if scores else 0.0


def _system_metrics(
    system_name: str,
    cases_by_id: dict[str, EvaluationCase],
    predictions: list[PredictionRecord],
) -> SystemMetrics:
    prediction_count = len(predictions)
    initial_correct = sum(
        prediction.predicted_initial_action
        == cases_by_id[prediction.case_id].expected_initial_action
        for prediction in predictions
    )
    final_scored = [
        prediction
        for prediction in predictions
        if cases_by_id[prediction.case_id].expected_final_action is not None
    ]
    final_correct = sum(
        prediction.predicted_final_action
        == cases_by_id[prediction.case_id].expected_final_action
        for prediction in final_scored
    )
    per_action = _action_metrics(cases_by_id, predictions)

    accuracy_by_difficulty: dict[str, float] = {}
    for difficulty in Difficulty:
        subset = [
            prediction
            for prediction in predictions
            if cases_by_id[prediction.case_id].difficulty == difficulty
        ]
        correct = sum(
            prediction.predicted_initial_action
            == cases_by_id[prediction.case_id].expected_initial_action
            for prediction in subset
        )
        accuracy_by_difficulty[difficulty.value] = _safe_ratio(correct, len(subset))

    durations = [
        prediction.duration_seconds
        for prediction in predictions
        if prediction.duration_seconds is not None
    ]
    return SystemMetrics(
        system_name=system_name,
        prediction_count=prediction_count,
        case_coverage=_safe_ratio(
            len({prediction.case_id for prediction in predictions}),
            len(cases_by_id),
        ),
        error_rate=_safe_ratio(
            sum(prediction.error_message is not None for prediction in predictions),
            prediction_count,
        ),
        initial_action_accuracy=_safe_ratio(initial_correct, prediction_count),
        final_action_accuracy=(
            _safe_ratio(final_correct, len(final_scored)) if final_scored else None
        ),
        macro_f1=mean(metric.f1 for metric in per_action.values()),
        action_metrics=per_action,
        accuracy_by_difficulty=accuracy_by_difficulty,
        mean_retries=(
            mean(prediction.retries_used for prediction in predictions)
            if predictions
            else 0.0
        ),
        mean_duration_seconds=mean(durations) if durations else None,
        median_duration_seconds=median(durations) if durations else None,
        mean_repeat_consistency=_repeat_consistency(predictions),
    )


def evaluate_predictions(
    cases: list[EvaluationCase],
    predictions: list[PredictionRecord],
) -> EvaluationReport:
    """Validate and score predictions from one or more named systems."""

    if not cases:
        raise EvaluationError("at least one evaluation case is required")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise EvaluationError("evaluation case IDs must be unique")
    cases_by_id = {case.case_id: case for case in cases}

    seen_predictions: set[tuple[str, str, int]] = set()
    by_system: dict[str, list[PredictionRecord]] = defaultdict(list)
    for prediction in predictions:
        if prediction.case_id not in cases_by_id:
            raise EvaluationError(
                f"prediction references unknown case: {prediction.case_id}"
            )
        key = (prediction.system_name, prediction.case_id, prediction.repetition)
        if key in seen_predictions:
            raise EvaluationError(
                "duplicate prediction for system, case, and repetition: "
                f"{key}"
            )
        seen_predictions.add(key)
        by_system[prediction.system_name].append(prediction)

    if not by_system:
        raise EvaluationError("at least one prediction is required")

    provenance_counts = Counter(case.label_provenance.value for case in cases)
    scientifically_valid = all(
        case.label_provenance == LabelProvenance.EXPERT_ANNOTATED for case in cases
    )
    return EvaluationReport(
        generated_at=datetime.now(timezone.utc),
        case_count=len(cases),
        scientifically_valid=scientifically_valid,
        label_provenance_counts=dict(provenance_counts),
        systems={
            system_name: _system_metrics(system_name, cases_by_id, system_predictions)
            for system_name, system_predictions in sorted(by_system.items())
        },
    )


def prediction_from_run_record(
    record: RunRecord,
    *,
    case_id: str,
    system_name: str,
    repetition: int,
    duration_seconds: float | None = None,
) -> PredictionRecord:
    """Normalize a successful workflow audit record for evaluation."""

    if not record.final_state.decision_history:
        raise EvaluationError("run record has no decision history")
    final_decision = record.final_state.decision
    if final_decision is None or final_decision.action == DecisionAction.RETRY:
        raise EvaluationError("run record does not contain a terminal decision")

    return PredictionRecord(
        case_id=case_id,
        system_name=system_name,
        repetition=repetition,
        run_id=record.run_id,
        predicted_initial_action=record.final_state.decision_history[0].action,
        predicted_final_action=final_decision.action.value,
        retries_used=record.final_state.retries_used,
        duration_seconds=duration_seconds,
    )


def write_evaluation_report(
    report: EvaluationReport,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Serialize an evaluation report without silently replacing prior results."""

    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite evaluation report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "ActionMetrics",
    "Difficulty",
    "EvaluationCase",
    "EvaluationError",
    "EvaluationReport",
    "LabelProvenance",
    "PredictionRecord",
    "SystemMetrics",
    "evaluate_predictions",
    "prediction_from_run_record",
    "write_evaluation_report",
]

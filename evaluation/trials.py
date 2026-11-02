"""Repeated synthetic model trials for consistency and runtime measurement."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent.config import AgentConfig, EXAMPLE_CONFIG_PATH, load_config
from agent.run import (
    RunRecord,
    ScenarioName,
    build_synthetic_initial_state,
    run_synthetic_case,
)
from agent.state import DecisionAction, RegionState
from evaluation.metrics import (
    Difficulty,
    EvaluationCase,
    EvaluationReport,
    LabelProvenance,
    PredictionRecord,
    evaluate_predictions,
    prediction_from_run_record,
)


class TrialBatchError(ValueError):
    """Raised when a repeated-trial batch is configured incorrectly."""


class ConsistencyTrialBatch(BaseModel):
    """Complete record of repeated assessments of one identical synthetic state."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str
    scenario: ScenarioName
    repetitions_requested: int = Field(ge=2)
    input_state_sha256: str
    model_provider: str
    model_name: str
    run_records: list[RunRecord]
    predictions: list[PredictionRecord]
    decision_distribution: dict[str, int]
    evaluation: EvaluationReport


def _case_for_scenario(scenario: ScenarioName) -> EvaluationCase:
    expected = {
        "accept": (DecisionAction.ACCEPT, "accept", Difficulty.EASY),
        "escalate": (DecisionAction.ESCALATE, "escalate", Difficulty.HARD),
        "retry-success": (DecisionAction.RETRY, "accept", Difficulty.HARD),
        "retry-exhausted": (DecisionAction.RETRY, "escalate", Difficulty.HARD),
    }
    initial_action, final_action, difficulty = expected[scenario]
    return EvaluationCase(
        case_id=f"synthetic-{scenario}",
        expected_initial_action=initial_action,
        expected_final_action=final_action,
        difficulty=difficulty,
        label_provenance=LabelProvenance.SYNTHETIC,
    )


def _state_fingerprint(state: RegionState) -> str:
    payload = state.model_dump_json(exclude_computed_fields=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_model_consistency_trials(
    *,
    config: AgentConfig,
    scenario: ScenarioName,
    repetitions: int,
) -> ConsistencyTrialBatch:
    """Run the real configured model repeatedly on one identical synthetic state."""

    if repetitions < 2:
        raise TrialBatchError("consistency trials require at least two repetitions")

    fixed_state = build_synthetic_initial_state(scenario)
    case = _case_for_scenario(scenario)
    system_name = f"model:{config.model.provider}:{config.model.name}"
    run_records: list[RunRecord] = []
    predictions: list[PredictionRecord] = []

    for repetition in range(1, repetitions + 1):
        started = perf_counter()
        try:
            record = run_synthetic_case(
                config=config,
                scenario=scenario,
                assessor_kind="model",
                initial_state=fixed_state,
            )
        except Exception as exc:
            duration = perf_counter() - started
            predictions.append(
                PredictionRecord(
                    case_id=case.case_id,
                    system_name=system_name,
                    repetition=repetition,
                    run_id=f"failed-{uuid4()}",
                    predicted_initial_action=None,
                    predicted_final_action=None,
                    retries_used=0,
                    duration_seconds=duration,
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        duration = perf_counter() - started
        run_records.append(record)
        predictions.append(
            prediction_from_run_record(
                record,
                case_id=case.case_id,
                system_name=system_name,
                repetition=repetition,
                duration_seconds=duration,
            )
        )

    distribution = Counter(
        prediction.predicted_initial_action.value
        if prediction.predicted_initial_action is not None
        else "error"
        for prediction in predictions
    )
    return ConsistencyTrialBatch(
        batch_id=str(uuid4()),
        scenario=scenario,
        repetitions_requested=repetitions,
        input_state_sha256=_state_fingerprint(fixed_state),
        model_provider=config.model.provider,
        model_name=config.model.name,
        run_records=run_records,
        predictions=predictions,
        decision_distribution=dict(distribution),
        evaluation=evaluate_predictions([case], predictions),
    )


def write_trial_batch(
    batch: ConsistencyTrialBatch,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a batch audit log without silently replacing an earlier experiment."""

    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite trial batch: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        batch.model_dump_json(indent=2, exclude_computed_fields=True) + "\n",
        encoding="utf-8",
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repeat one identical synthetic case against the configured real model."
        )
    )
    parser.add_argument("--config", default=str(EXAMPLE_CONFIG_PATH))
    parser.add_argument(
        "--scenario",
        choices=("accept", "escalate", "retry-success", "retry-exhausted"),
        default="escalate",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        help="Override evaluation.consistency_repetitions for this run.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    batch = run_model_consistency_trials(
        config=config,
        scenario=args.scenario,
        repetitions=(
            config.evaluation.consistency_repetitions
            if args.repetitions is None
            else args.repetitions
        ),
    )
    output = args.output or (
        config.paths.output_directory / f"consistency-{batch.batch_id}.json"
    )
    path = write_trial_batch(batch, output, overwrite=args.overwrite)
    metrics = next(iter(batch.evaluation.systems.values()))
    print(f"batch: {batch.batch_id}")
    print(f"input_state_sha256: {batch.input_state_sha256}")
    print(f"decisions: {batch.decision_distribution}")
    print(f"repeat_consistency: {metrics.mean_repeat_consistency:.3f}")
    print(f"error_rate: {metrics.error_rate:.3f}")
    print(f"output: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ConsistencyTrialBatch",
    "TrialBatchError",
    "run_model_consistency_trials",
    "write_trial_batch",
]

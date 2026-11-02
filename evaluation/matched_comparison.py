"""Matched synthetic comparison of the model-backed and static assessors.

Everything declared here is a workflow mechanics fixture. The rules, labels, and
retry strategy are synthetic and must not be interpreted as biological guidance.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from time import perf_counter
from typing import Literal, Sequence
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from agent.config import AgentConfig, EXAMPLE_CONFIG_PATH, load_config
from agent.llm import ModelAssessor, create_chat_model
from agent.nodes import Assessor
from agent.pipeline_interface import MockPhasingBackend
from agent.run import (
    RunRecord,
    ScenarioName,
    build_synthetic_initial_state,
    run_synthetic_case_with_assessor,
)
from agent.state import (
    DecisionAction,
    PhasingAttempt,
    PhasingEvidence,
    PhasingOutcome,
    PhasingStrategy,
    RegionState,
)
from agent.static_baseline import (
    PredicateRule,
    RuleProvenance,
    StaticBaselineAssessor,
    StaticRuleDecision,
)
from evaluation.metrics import (
    Difficulty,
    EvaluationCase,
    EvaluationReport,
    LabelProvenance,
    PredictionRecord,
    evaluate_predictions,
    prediction_from_run_record,
)


COMPARISON_SCENARIOS: tuple[ScenarioName, ...] = (
    "accept",
    "retry-success",
    "escalate",
)
MECHANICS_RETRY_STRATEGY = PhasingStrategy(
    strategy_id="synthetic-mechanics-retry-unvalidated",
    parameters={
        "test_fixture": True,
        "validation_status": "unvalidated",
        "scope": "mechanics-test-only",
    },
)


class MatchedComparisonError(ValueError):
    """Raised when a matched comparison is configured incorrectly."""


class MatchedComparisonBatch(BaseModel):
    """One auditable, explicitly non-scientific two-system comparison."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    batch_id: str = Field(min_length=1)
    data_status: Literal["synthetic"] = "synthetic"
    validation_status: Literal["mechanics_test_only_unvalidated"] = (
        "mechanics_test_only_unvalidated"
    )
    scientific_interpretation_allowed: Literal[False] = False
    repetitions_per_case: int = Field(ge=2)
    cases: list[EvaluationCase]
    input_state_sha256: dict[str, str]
    retry_strategy: PhasingStrategy
    run_records: list[RunRecord]
    predictions: list[PredictionRecord]
    evaluation: EvaluationReport


def _state_fingerprint(state: RegionState) -> str:
    payload = state.model_dump_json(exclude_computed_fields=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _comparison_cases() -> list[EvaluationCase]:
    """Declare fake labels used only to verify comparison mechanics."""

    return [
        EvaluationCase(
            case_id="synthetic-accept",
            expected_initial_action=DecisionAction.ACCEPT,
            expected_final_action="accept",
            difficulty=Difficulty.EASY,
            label_provenance=LabelProvenance.SYNTHETIC,
        ),
        EvaluationCase(
            case_id="synthetic-retry-success",
            expected_initial_action=DecisionAction.RETRY,
            expected_final_action="accept",
            difficulty=Difficulty.HARD,
            label_provenance=LabelProvenance.SYNTHETIC,
        ),
        EvaluationCase(
            case_id="synthetic-escalate",
            expected_initial_action=DecisionAction.ESCALATE,
            expected_final_action="escalate",
            difficulty=Difficulty.HARD,
            label_provenance=LabelProvenance.SYNTHETIC,
        ),
    ]


def _comparison_state(scenario: ScenarioName, case_number: int) -> RegionState:
    """Build a fake input without exposing its expected action to the assessor."""

    state = build_synthetic_initial_state(scenario)
    if scenario == "retry-success":
        evidence = PhasingEvidence(
            outcome=PhasingOutcome.PARTIALLY_PHASED,
            heterozygous_snp_count=4,
            phased_snp_count=1,
            unphased_snp_count=3,
            haplotype_count=2,
            tool_messages=[
                "Synthetic comparison fixture; no biological interpretation."
            ],
        )
    else:
        evidence = state.evidence.model_copy(deep=True)
        evidence.tool_messages = [
            "Synthetic comparison fixture; no biological interpretation."
        ]
    baseline = PhasingAttempt(
        attempt_number=1,
        strategy_id="synthetic-baseline",
        evidence=evidence,
    )
    return state.model_copy(
        update={
            "evidence": evidence,
            "attempt_history": [baseline],
            "metadata": {"fixture": True, "case_code": f"matched-{case_number:03d}"},
        },
        deep=True,
    )


def build_mechanics_only_static_assessor(
    config: AgentConfig,
) -> StaticBaselineAssessor:
    """Build fake rules that deliberately cover all graph routes."""

    definition = config.static_baseline.model_copy(
        update={
            "name": "synthetic-mechanics-static-baseline",
            "version": "0.1.0-unvalidated",
        }
    )
    rules = [
        PredicateRule(
            rule_id="synthetic-phased-accept",
            provenance=RuleProvenance.SYNTHETIC_ONLY,
            predicate=lambda state: state.evidence.outcome == PhasingOutcome.PHASED,
            decision=StaticRuleDecision(
                action=DecisionAction.ACCEPT,
                confidence=1.0,
                rationale="Synthetic mechanics fixture: phased outcome accepts.",
            ),
        ),
        PredicateRule(
            rule_id="synthetic-retry-once",
            provenance=RuleProvenance.SYNTHETIC_ONLY,
            predicate=lambda state: (
                state.evidence.outcome == PhasingOutcome.PARTIALLY_PHASED
                and MECHANICS_RETRY_STRATEGY.strategy_id
                not in state.strategies_tried
            ),
            decision=StaticRuleDecision(
                action=DecisionAction.RETRY,
                confidence=1.0,
                rationale=(
                    "Synthetic mechanics fixture: exercise one retry; this is not "
                    "a biologically approved rule."
                ),
                retry_strategy=MECHANICS_RETRY_STRATEGY,
            ),
        ),
        PredicateRule(
            rule_id="synthetic-failed-escalate",
            provenance=RuleProvenance.SYNTHETIC_ONLY,
            predicate=lambda state: state.evidence.outcome == PhasingOutcome.FAILED,
            decision=StaticRuleDecision(
                action=DecisionAction.ESCALATE,
                confidence=1.0,
                rationale="Synthetic mechanics fixture: failed outcome escalates.",
            ),
        ),
    ]
    return StaticBaselineAssessor(
        definition=definition,
        rules=rules,
        approved_retry_strategies=[MECHANICS_RETRY_STRATEGY],
    )


def _comparison_config(config: AgentConfig) -> AgentConfig:
    """Add the isolated mechanics strategy to a run copy, never canonical config."""

    pipeline = config.pipeline.model_copy(
        update={"retry_strategies": [MECHANICS_RETRY_STRATEGY]}
    )
    return config.model_copy(update={"pipeline": pipeline})


def _backend_for_scenario(scenario: ScenarioName) -> MockPhasingBackend:
    retry_evidence = PhasingEvidence(
        outcome=(
            PhasingOutcome.PHASED
            if scenario == "retry-success"
            else PhasingOutcome.FAILED
        ),
        heterozygous_snp_count=4,
        phased_snp_count=4 if scenario == "retry-success" else 0,
        unphased_snp_count=0 if scenario == "retry-success" else 4,
        haplotype_count=2 if scenario == "retry-success" else None,
        tool_messages=[
            "Synthetic mechanics-only retry result; no biological interpretation."
        ],
    )
    return MockPhasingBackend(
        {MECHANICS_RETRY_STRATEGY.strategy_id: retry_evidence}
    )


def _failed_prediction(
    *,
    case_id: str,
    system_name: str,
    repetition: int,
    duration_seconds: float,
    error: Exception,
) -> PredictionRecord:
    return PredictionRecord(
        case_id=case_id,
        system_name=system_name,
        repetition=repetition,
        run_id=f"failed-{uuid4()}",
        predicted_initial_action=None,
        predicted_final_action=None,
        retries_used=0,
        duration_seconds=duration_seconds,
        error_message=f"{type(error).__name__}: {error}",
    )


def run_matched_synthetic_comparison(
    *,
    config: AgentConfig,
    repetitions: int,
    model: BaseChatModel | None = None,
    model_assessor: Assessor | None = None,
) -> MatchedComparisonBatch:
    """Run both systems on identical states with identical repetition counts."""

    if repetitions < 2:
        raise MatchedComparisonError("matched comparison requires at least two repetitions")
    if model is not None and model_assessor is not None:
        raise MatchedComparisonError("provide model or model_assessor, not both")

    run_config = _comparison_config(config)
    if model_assessor is None:
        selected_model = model or create_chat_model(run_config.model)
        model_assessor = ModelAssessor.from_config(
            model=selected_model,
            config=run_config,
        )
    static_assessor = build_mechanics_only_static_assessor(run_config)

    cases = _comparison_cases()
    states = {
        scenario: _comparison_state(scenario, case_number)
        for case_number, scenario in enumerate(COMPARISON_SCENARIOS, start=1)
    }
    system_assessors: tuple[tuple[str, Literal["model", "static"], Assessor], ...] = (
        (
            f"model:{run_config.model.provider}:{run_config.model.name}",
            "model",
            model_assessor,
        ),
        (
            "static:synthetic-mechanics-static-baseline:0.1.0-unvalidated",
            "static",
            static_assessor,
        ),
    )
    run_records: list[RunRecord] = []
    predictions: list[PredictionRecord] = []

    for scenario, case in zip(COMPARISON_SCENARIOS, cases, strict=True):
        for repetition in range(1, repetitions + 1):
            for system_name, assessor_kind, assessor in system_assessors:
                started = perf_counter()
                try:
                    record = run_synthetic_case_with_assessor(
                        config=run_config,
                        scenario=scenario,
                        assessor_kind=assessor_kind,
                        assessor=assessor,
                        backend=_backend_for_scenario(scenario),
                        initial_state=states[scenario],
                        model_provider=(
                            run_config.model.provider
                            if assessor_kind == "model"
                            else None
                        ),
                        model_name=(
                            run_config.model.name if assessor_kind == "model" else None
                        ),
                    )
                except Exception as exc:
                    predictions.append(
                        _failed_prediction(
                            case_id=case.case_id,
                            system_name=system_name,
                            repetition=repetition,
                            duration_seconds=perf_counter() - started,
                            error=exc,
                        )
                    )
                    continue

                run_records.append(record)
                predictions.append(
                    prediction_from_run_record(
                        record,
                        case_id=case.case_id,
                        system_name=system_name,
                        repetition=repetition,
                        duration_seconds=perf_counter() - started,
                    )
                )

    return MatchedComparisonBatch(
        batch_id=str(uuid4()),
        repetitions_per_case=repetitions,
        cases=cases,
        input_state_sha256={
            f"synthetic-{scenario}": _state_fingerprint(states[scenario])
            for scenario in COMPARISON_SCENARIOS
        },
        retry_strategy=MECHANICS_RETRY_STRATEGY,
        run_records=run_records,
        predictions=predictions,
        evaluation=evaluate_predictions(cases, predictions),
    )


def write_matched_comparison(
    batch: MatchedComparisonBatch,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one shared comparison report without silent replacement."""

    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite comparison report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        batch.model_dump_json(indent=2, exclude_computed_fields=True) + "\n",
        encoding="utf-8",
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the non-biological matched synthetic comparison."
    )
    parser.add_argument("--config", default=str(EXAMPLE_CONFIG_PATH))
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
    repetitions = (
        config.evaluation.consistency_repetitions
        if args.repetitions is None
        else args.repetitions
    )
    batch = run_matched_synthetic_comparison(
        config=config,
        repetitions=repetitions,
    )
    output = args.output or (
        config.paths.output_directory / f"matched-synthetic-{batch.batch_id}.json"
    )
    path = write_matched_comparison(batch, output, overwrite=args.overwrite)

    print(f"batch: {batch.batch_id}")
    print(f"status: {batch.validation_status}")
    for name, metrics in batch.evaluation.systems.items():
        print(
            f"{name}: predictions={metrics.prediction_count}, "
            f"errors={metrics.error_rate:.3f}, "
            f"synthetic_initial_accuracy={metrics.initial_action_accuracy:.3f}, "
            f"repeat_consistency={metrics.mean_repeat_consistency:.3f}"
        )
    print(f"scientific_interpretation_allowed: false")
    print(f"output: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPARISON_SCENARIOS",
    "MECHANICS_RETRY_STRATEGY",
    "MatchedComparisonBatch",
    "MatchedComparisonError",
    "build_mechanics_only_static_assessor",
    "run_matched_synthetic_comparison",
    "write_matched_comparison",
]

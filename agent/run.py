"""Command-line entry point for synthetic Haplotype Mapper workflow runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent.config import AgentConfig, EXAMPLE_CONFIG_PATH, PipelineMode, load_config
from agent.graph import build_agent_graph
from agent.llm import ModelAssessor, create_chat_model
from agent.nodes import AssessmentResult, Assessor
from agent.pipeline_interface import MockPhasingBackend, PipelineInterface
from agent.state import (
    DecisionAction,
    DecisionSource,
    PhasingAttempt,
    PhasingEvidence,
    PhasingOutcome,
    PhasingStrategy,
    RegionIdentity,
    RegionState,
)


ScenarioName = Literal[
    "accept",
    "escalate",
    "retry-success",
    "retry-exhausted",
]
AssessorKind = Literal["scripted", "model", "static"]


class RunRecord(BaseModel):
    """Serializable audit record for one explicitly synthetic workflow run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    recorded_at: datetime
    data_mode: Literal["synthetic"] = "synthetic"
    scientific_interpretation_allowed: Literal[False] = False
    scenario: ScenarioName
    assessor_kind: AssessorKind
    model_provider: str | None
    model_name: str | None
    configuration: dict[str, object]
    initial_state: RegionState
    final_state: RegionState


class ScriptedAssessor:
    """Fixed synthetic decisions used to test orchestration without inference."""

    def __init__(self, assessments: Sequence[AssessmentResult]) -> None:
        self._assessments = iter(assessments)

    def assess(self, state: RegionState) -> AssessmentResult:
        try:
            return next(self._assessments)
        except StopIteration as exc:
            raise RuntimeError("synthetic assessor ran out of scripted decisions") from exc


class RetryUntilGuardedAssessor:
    """Propose unique synthetic retries until graph safeguards escalate."""

    def assess(self, state: RegionState) -> AssessmentResult:
        next_number = len(state.attempt_history) + 1
        return AssessmentResult(
            action=DecisionAction.RETRY,
            confidence=0.5,
            rationale="Synthetic retry-exhaustion scenario.",
            source=DecisionSource.DETERMINISTIC,
            retry_strategy=PhasingStrategy(
                strategy_id=f"synthetic-retry-{next_number}"
            ),
        )


def _assessment(
    action: DecisionAction,
    rationale: str,
    *,
    retry_strategy: PhasingStrategy | None = None,
) -> AssessmentResult:
    return AssessmentResult(
        action=action,
        confidence=1.0,
        rationale=rationale,
        source=DecisionSource.DETERMINISTIC,
        retry_strategy=retry_strategy,
    )


def build_synthetic_initial_state(scenario: ScenarioName) -> RegionState:
    """Build one explicitly fake baseline state for repeatable preparation tests."""

    baseline_outcome = (
        PhasingOutcome.PHASED
        if scenario == "accept"
        else PhasingOutcome.FAILED
    )
    baseline_evidence = PhasingEvidence(
        outcome=baseline_outcome,
        heterozygous_snp_count=4,
        phased_snp_count=4 if baseline_outcome == PhasingOutcome.PHASED else 0,
        unphased_snp_count=0 if baseline_outcome == PhasingOutcome.PHASED else 4,
        haplotype_count=2 if baseline_outcome == PhasingOutcome.PHASED else None,
        tool_messages=["Synthetic fixture: no biological interpretation."],
    )
    baseline = PhasingAttempt(
        attempt_number=1,
        strategy_id="synthetic-baseline",
        evidence=baseline_evidence,
    )
    return RegionState(
        sample_id="SYNTHETIC_SAMPLE",
        region=RegionIdentity(
            gene="SYNTHETIC_GENE",
            target_region="Synthetic target region",
            chromosome="ChrSynthetic",
            start_position=100,
            end_position=120,
        ),
        evidence=baseline_evidence,
        attempt_history=[baseline],
        warnings=["Synthetic case: not suitable for biological interpretation."],
        metadata={"fixture": True, "scenario": scenario},
    )


def _build_synthetic_case(
    scenario: ScenarioName,
    config: AgentConfig,
) -> tuple[RegionState, MockPhasingBackend, Assessor]:
    """Build labelled fake evidence, backend responses, and scripted decisions."""

    initial_state = build_synthetic_initial_state(scenario)

    if scenario == "accept":
        assessor: Assessor = ScriptedAssessor(
            [_assessment(DecisionAction.ACCEPT, "Synthetic acceptance scenario.")]
        )
        results: dict[str, PhasingEvidence] = {}
    elif scenario == "escalate":
        assessor = ScriptedAssessor(
            [_assessment(DecisionAction.ESCALATE, "Synthetic escalation scenario.")]
        )
        results = {}
    elif scenario == "retry-success":
        strategy = PhasingStrategy(strategy_id="synthetic-retry-success")
        assessor = ScriptedAssessor(
            [
                _assessment(
                    DecisionAction.RETRY,
                    "Synthetic recoverable failure.",
                    retry_strategy=strategy,
                ),
                _assessment(
                    DecisionAction.ACCEPT,
                    "Synthetic retry produced the configured success result.",
                ),
            ]
        )
        results = {
            strategy.strategy_id: PhasingEvidence(
                outcome=PhasingOutcome.PHASED,
                heterozygous_snp_count=4,
                phased_snp_count=4,
                unphased_snp_count=0,
                haplotype_count=2,
                tool_messages=["Synthetic retry result."],
            )
        }
    else:
        assessor = RetryUntilGuardedAssessor()
        results = {
            f"synthetic-retry-{number}": PhasingEvidence(
                outcome=PhasingOutcome.FAILED,
                tool_messages=["Synthetic failed retry."],
            )
            for number in range(2, config.decisions.max_retries + 2)
        }

    return initial_state, MockPhasingBackend(results), assessor


def _mock_results_for_model(
    scenario: ScenarioName,
    config: AgentConfig,
) -> MockPhasingBackend:
    outcome = (
        PhasingOutcome.PHASED
        if scenario == "retry-success"
        else PhasingOutcome.FAILED
    )
    return MockPhasingBackend(
        {
            strategy.strategy_id: PhasingEvidence(
                outcome=outcome,
                tool_messages=["Synthetic model-assessor backend result."],
            )
            for strategy in config.pipeline.retry_strategies
        }
    )


def run_synthetic_case_with_assessor(
    *,
    config: AgentConfig,
    scenario: ScenarioName,
    assessor_kind: AssessorKind,
    assessor: Assessor,
    backend: MockPhasingBackend,
    initial_state: RegionState,
    model_provider: str | None = None,
    model_name: str | None = None,
) -> RunRecord:
    """Run one synthetic state with injected comparison components."""

    if config.pipeline.mode != PipelineMode.MOCK:
        raise ValueError("synthetic runs require pipeline.mode=mock")

    graph = build_agent_graph(
        config=config,
        assessor=assessor,
        pipeline=PipelineInterface(backend, validate_inputs=False),
    )
    recursion_limit = config.execution.graph_recursion_limit(
        config.decisions.max_retries
    )
    raw_final_state = graph.invoke(
        initial_state.model_copy(deep=True),
        config={"recursion_limit": recursion_limit},
    )
    final_state = RegionState.model_validate(raw_final_state)

    return RunRecord(
        run_id=str(uuid4()),
        recorded_at=datetime.now(timezone.utc),
        scenario=scenario,
        assessor_kind=assessor_kind,
        model_provider=model_provider,
        model_name=model_name,
        configuration=config.model_dump(mode="json"),
        initial_state=initial_state,
        final_state=final_state,
    )


def run_synthetic_case(
    *,
    config: AgentConfig,
    scenario: ScenarioName,
    assessor_kind: AssessorKind,
    initial_state: RegionState | None = None,
) -> RunRecord:
    """Execute one synthetic graph scenario and return its complete audit record."""

    if config.pipeline.mode != PipelineMode.MOCK:
        raise ValueError("synthetic runs require pipeline.mode=mock")

    generated_state, scripted_backend, scripted_assessor = _build_synthetic_case(
        scenario, config
    )
    case_state = initial_state or generated_state
    if assessor_kind == "scripted":
        assessor = scripted_assessor
        backend = scripted_backend
        model_provider = None
        model_name = None
    else:
        model = create_chat_model(config.model)
        assessor = ModelAssessor.from_config(model=model, config=config)
        backend = _mock_results_for_model(scenario, config)
        model_provider = config.model.provider
        model_name = config.model.name

    return run_synthetic_case_with_assessor(
        config=config,
        scenario=scenario,
        assessor_kind=assessor_kind,
        assessor=assessor,
        backend=backend,
        initial_state=case_state,
        model_provider=model_provider,
        model_name=model_name,
    )


def write_run_record(
    record: RunRecord,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write an audit record without silently replacing an existing result."""

    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing run record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        record.model_dump_json(indent=2, exclude_computed_fields=True) + "\n",
        encoding="utf-8",
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a clearly labelled synthetic Haplotype Mapper scenario."
    )
    parser.add_argument(
        "--config",
        default=str(EXAMPLE_CONFIG_PATH),
        help="Path to agent YAML configuration.",
    )
    parser.add_argument(
        "--scenario",
        choices=("accept", "escalate", "retry-success", "retry-exhausted"),
        default="accept",
    )
    parser.add_argument(
        "--assessor",
        choices=("scripted", "model"),
        default="scripted",
        help="Model inference is opt-in; scripted is the safe default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path; defaults to outputs/<run-id>.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit replacement of the explicitly selected output file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    record = run_synthetic_case(
        config=config,
        scenario=args.scenario,
        assessor_kind=args.assessor,
    )
    output_path = args.output or (
        config.paths.output_directory / f"synthetic-{record.run_id}.json"
    )
    written_path = write_run_record(
        record,
        output_path,
        overwrite=args.overwrite,
    )
    print(
        f"synthetic run {record.run_id}: "
        f"{record.final_state.decision.action.value} -> {written_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AssessorKind",
    "RunRecord",
    "ScenarioName",
    "build_synthetic_initial_state",
    "run_synthetic_case",
    "run_synthetic_case_with_assessor",
    "write_run_record",
]

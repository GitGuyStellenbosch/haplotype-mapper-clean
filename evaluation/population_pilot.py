"""Non-scoring model pilot for population-frequency-simulated genotypes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.config import AgentConfig, EXAMPLE_CONFIG_PATH, PipelineMode, load_config
from agent.graph import build_agent_graph
from agent.llm import ModelAssessor, create_chat_model
from agent.nodes import Assessor
from agent.pipeline_interface import MockPhasingBackend, PipelineInterface
from agent.state import RegionState
from evaluation.population_simulation import (
    DEFAULT_INDIVIDUAL_COUNT,
    DEFAULT_PILOT_PER_STRATUM,
    DEFAULT_SIMULATION_SEED,
    POPULATION_SIMULATION_DATA_STATUS,
    PopulationDatasetManifest,
    PopulationSimulationCase,
    SampledGenotypeTruth,
    generate_population_dataset,
    load_population_cases,
)


class PopulationPilotError(RuntimeError):
    """Raised when a simulation pilot cannot run or resume safely."""


class PilotModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PopulationPilotAudit(PilotModel):
    """One model result that cannot be represented as an accuracy record."""

    audit_schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    recorded_at: datetime
    data_status: Literal[
        "simulated_from_population_frequencies"
    ] = POPULATION_SIMULATION_DATA_STATUS
    truth_level: Literal["genotype"] = "genotype"
    action_ground_truth_available: Literal[False] = False
    scoring_performed: Literal[False] = False
    scientific_interpretation_allowed: Literal[False] = False
    execution_status: Literal["success", "error"]
    case_id: str = Field(min_length=1)
    model_provider: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    duration_seconds: float = Field(ge=0.0)
    genotype_truth: SampledGenotypeTruth
    initial_state: RegionState
    final_state: RegionState | None = None
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def result_and_error_are_consistent(self) -> "PopulationPilotAudit":
        if self.execution_status == "success":
            if self.final_state is None:
                raise ValueError("successful pilot audit requires final_state")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful pilot audit cannot contain an error")
        else:
            if self.final_state is not None:
                raise ValueError("failed pilot audit cannot contain final_state")
            if not self.error_type or not self.error_message:
                raise ValueError("failed pilot audit requires error details")
        if self.initial_state.metadata.get("data_status") != self.data_status:
            raise ValueError("pilot state does not carry the simulation status")
        return self


class ManualReviewItem(PilotModel):
    """A retry/escalate result highlighted without asserting it is incorrect."""

    case_id: str
    catalogue_category: Literal[1, 2, 3]
    zygosity: Literal["homozygous", "heterozygous"]
    heterozygous_snp_count: int = Field(ge=0)
    action: Literal["retry", "escalate"]
    rationale: str


class PopulationPilotReport(PilotModel):
    """Descriptive aggregate of one stratified development pilot."""

    report_schema_version: Literal["1.0"] = "1.0"
    batch_id: str
    generated_at: datetime
    data_status: Literal[
        "simulated_from_population_frequencies"
    ] = POPULATION_SIMULATION_DATA_STATUS
    truth_level: Literal["genotype"] = "genotype"
    action_ground_truth_available: Literal[False] = False
    scoring_performed: Literal[False] = False
    accuracy: None = None
    partition: Literal["development"] = "development"
    requested_cases: int = Field(ge=1)
    processed_cases: int = Field(ge=0)
    successful_runs: int = Field(ge=0)
    error_count: int = Field(ge=0)
    action_counts: dict[str, int]
    decisions_by_stratum: dict[str, dict[str, int]]
    manual_review_non_accept: list[ManualReviewItem]
    example_rationales: list[str]
    interpretation_boundary: str = (
        "Descriptive decisions against genotype/zygosity only; no action accuracy."
    )


def _audit_filename(case_id: str) -> str:
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:16]
    return f"case-{digest}.json"


def _write_model(path: Path, value: BaseModel) -> None:
    path.write_text(
        value.model_dump_json(indent=2, exclude_computed_fields=True) + "\n",
        encoding="utf-8",
    )


def _load_audit(path: Path, case_id: str) -> PopulationPilotAudit:
    try:
        audit = PopulationPilotAudit.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PopulationPilotError(f"cannot resume invalid pilot audit: {path}") from exc
    if audit.case_id != case_id:
        raise PopulationPilotError(f"pilot audit case mismatch: {path}")
    return audit


def build_pilot_report(
    *,
    batch_id: str,
    requested_cases: int,
    records: Sequence[PopulationPilotAudit],
) -> PopulationPilotReport:
    """Aggregate decisions by category/zygosity without creating action labels."""

    action_counts: Counter[str] = Counter()
    by_stratum: dict[str, Counter[str]] = {}
    manual_review: list[ManualReviewItem] = []
    rationales: list[str] = []
    for record in records:
        if record.execution_status != "success" or record.final_state is None:
            continue
        if not record.final_state.decision_history:
            continue
        decision = record.final_state.decision_history[0]
        action = decision.action.value
        action_counts[action] += 1
        truth = record.genotype_truth
        stratum = f"category_{truth.catalogue_category}:{truth.zygosity}"
        by_stratum.setdefault(stratum, Counter())[action] += 1
        if len(rationales) < 6:
            rationales.append(f"{record.case_id}: [{action}] {decision.rationale}")
        if action in {"retry", "escalate"}:
            manual_review.append(
                ManualReviewItem(
                    case_id=record.case_id,
                    catalogue_category=truth.catalogue_category,
                    zygosity=truth.zygosity,
                    heterozygous_snp_count=truth.heterozygous_snp_count,
                    action=action,
                    rationale=decision.rationale,
                )
            )

    errors = sum(record.execution_status == "error" for record in records)
    return PopulationPilotReport(
        batch_id=batch_id,
        generated_at=datetime.now(timezone.utc),
        requested_cases=requested_cases,
        processed_cases=len(records),
        successful_runs=len(records) - errors,
        error_count=errors,
        action_counts={action: action_counts[action] for action in ("accept", "retry", "escalate")},
        decisions_by_stratum={
            key: {action: counts[action] for action in ("accept", "retry", "escalate")}
            for key, counts in sorted(by_stratum.items())
        },
        manual_review_non_accept=manual_review,
        example_rationales=rationales,
    )


def render_pilot_summary(report: PopulationPilotReport) -> str:
    """Render a concise human-readable, explicitly non-scoring report."""

    lines = [
        "Haplotype Mapper population-frequency simulation pilot",
        f"Batch: {report.batch_id}",
        f"Data status: {report.data_status}",
        "Partition: development",
        "Truth available: sampled haplotype IDs and zygosity",
        "Action ground truth available: no",
        "Scoring performed: no",
        "Accuracy: not calculated",
        "",
        f"Cases processed: {report.processed_cases}/{report.requested_cases}",
        f"Successful graph runs: {report.successful_runs}",
        f"Errors or crashes: {report.error_count}",
        "Decisions:",
        f"accept: {report.action_counts.get('accept', 0)}",
        f"retry: {report.action_counts.get('retry', 0)}",
        f"escalate: {report.action_counts.get('escalate', 0)}",
        "",
        "Decisions by catalogue category and sampled zygosity:",
    ]
    for stratum, counts in report.decisions_by_stratum.items():
        lines.append(
            f"- {stratum}: accept={counts.get('accept', 0)}, "
            f"retry={counts.get('retry', 0)}, escalate={counts.get('escalate', 0)}"
        )
    lines.extend(
        [
            "",
            f"Non-accept decisions flagged for manual review: {len(report.manual_review_non_accept)}",
        ]
    )
    for item in report.manual_review_non_accept:
        lines.append(f"- {item.case_id}: [{item.action}] {item.rationale}")
    if report.example_rationales:
        lines.extend(["", "Example rationales:"])
        lines.extend(f"- {rationale}" for rationale in report.example_rationales)
    lines.extend(["", f"Interpretation boundary: {report.interpretation_boundary}"])
    return "\n".join(lines) + "\n"


def _write_report(
    output: Path,
    *,
    batch_id: str,
    requested_cases: int,
    records: Sequence[PopulationPilotAudit],
) -> PopulationPilotReport:
    report = build_pilot_report(
        batch_id=batch_id,
        requested_cases=requested_cases,
        records=records,
    )
    _write_model(output / "pilot_report.json", report)
    (output / "pilot_summary.txt").write_text(
        render_pilot_summary(report), encoding="utf-8"
    )
    return report


def run_population_pilot(
    *,
    config: AgentConfig,
    cases: Sequence[PopulationSimulationCase],
    output_directory: str | Path,
    assessor: Assessor | None = None,
    resume: bool = False,
    progress_every: int = 1,
) -> PopulationPilotReport:
    """Run one model assessment per stratified development case."""

    if config.pipeline.mode != PipelineMode.MOCK:
        raise PopulationPilotError("genotype simulation pilot requires pipeline.mode=mock")
    if config.pipeline.retry_strategies:
        raise PopulationPilotError("simulation pilot requires an empty retry allowlist")
    if not cases:
        raise PopulationPilotError("simulation pilot requires at least one case")
    if progress_every < 1:
        raise PopulationPilotError("progress_every must be at least 1")
    if any(case.truth.partition != "development" for case in cases):
        raise PopulationPilotError("sanity pilot may use development cases only")

    output = Path(output_directory)
    audits_directory = output / "pilot_audits"
    if audits_directory.exists() and any(audits_directory.iterdir()) and not resume:
        raise FileExistsError(
            f"pilot audit directory is not empty; use resume: {audits_directory}"
        )
    audits_directory.mkdir(parents=True, exist_ok=True)
    selected_assessor = assessor
    if selected_assessor is None:
        model = create_chat_model(config.model)
        selected_assessor = ModelAssessor.from_config(model=model, config=config)
    graph = build_agent_graph(
        config=config,
        assessor=selected_assessor,
        pipeline=PipelineInterface(MockPhasingBackend({}), validate_inputs=False),
    )
    recursion_limit = config.execution.graph_recursion_limit(config.decisions.max_retries)
    batch_id = output.name
    records: list[PopulationPilotAudit] = []

    for case in cases:
        audit_path = audits_directory / _audit_filename(case.case_id)
        if audit_path.exists() and resume:
            records.append(_load_audit(audit_path, case.case_id))
            continue
        started = perf_counter()
        try:
            raw_final = graph.invoke(
                case.state.model_copy(deep=True),
                config={"recursion_limit": recursion_limit},
            )
            final_state = RegionState.model_validate(raw_final)
            record = PopulationPilotAudit(
                run_id=str(uuid4()),
                batch_id=batch_id,
                recorded_at=datetime.now(timezone.utc),
                execution_status="success",
                case_id=case.case_id,
                model_provider=config.model.provider,
                model_name=config.model.name,
                duration_seconds=perf_counter() - started,
                genotype_truth=case.truth,
                initial_state=case.state,
                final_state=final_state,
            )
        except Exception as exc:
            record = PopulationPilotAudit(
                run_id=str(uuid4()),
                batch_id=batch_id,
                recorded_at=datetime.now(timezone.utc),
                execution_status="error",
                case_id=case.case_id,
                model_provider=config.model.provider,
                model_name=config.model.name,
                duration_seconds=perf_counter() - started,
                genotype_truth=case.truth,
                initial_state=case.state,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        _write_model(audit_path, record)
        records.append(record)
        _write_report(
            output,
            batch_id=batch_id,
            requested_cases=len(cases),
            records=records,
        )
        if len(records) % progress_every == 0 or len(records) == len(cases):
            print(
                f"processed {len(records)}/{len(cases)} pilot cases; "
                f"latest={record.execution_status}; case={record.case_id}",
                flush=True,
            )
    return _write_report(
        output,
        batch_id=batch_id,
        requested_cases=len(cases),
        records=records,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate frequency-grounded genotypes and run a non-scoring pilot."
    )
    parser.add_argument("--config", default=str(EXAMPLE_CONFIG_PATH))
    parser.add_argument(
        "--workbook",
        default="data/ground_truth/candotti2023_haplotype_groundtruth.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"outputs/population-frequency-phase6-seed-{DEFAULT_SIMULATION_SEED}"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SIMULATION_SEED)
    parser.add_argument("--individuals", type=int, default=DEFAULT_INDIVIDUAL_COUNT)
    parser.add_argument(
        "--pilot-per-stratum", type=int, default=DEFAULT_PILOT_PER_STRATUM
    )
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.output_dir / "generation_manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"generation manifest already exists; use --resume: {manifest_path}"
            )
        manifest = PopulationDatasetManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    else:
        manifest = generate_population_dataset(
            workbook_path=args.workbook,
            output_directory=args.output_dir,
            seed=args.seed,
            individual_count=args.individuals,
            pilot_per_stratum=args.pilot_per_stratum,
        )
    print(
        f"generated cases={manifest.generated_case_count}; "
        f"excluded regions={len(manifest.excluded_regions)}; "
        f"pilot cases={manifest.pilot_case_count}; sha256={manifest.cases_sha256}",
        flush=True,
    )
    if args.generate_only:
        return 0
    cases = load_population_cases(manifest.pilot_cases_path)
    report = run_population_pilot(
        config=load_config(args.config),
        cases=cases,
        output_directory=args.output_dir,
        resume=args.resume,
        progress_every=args.progress_every,
    )
    print(render_pilot_summary(report))
    return 0 if report.error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ManualReviewItem",
    "PopulationPilotAudit",
    "PopulationPilotError",
    "PopulationPilotReport",
    "build_pilot_report",
    "render_pilot_summary",
    "run_population_pilot",
]

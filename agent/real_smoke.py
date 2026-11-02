"""Run the compiled agent graph on real, explicitly unlabeled VCF evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal, Mapping, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.config import AgentConfig, EXAMPLE_CONFIG_PATH, PipelineMode, load_config
from agent.graph import build_agent_graph
from agent.llm import ModelAssessmentError, ModelAssessor, create_chat_model, load_skills
from agent.real_retry import WhatsHapRetryBackend
from agent.pipeline_interface import (
    PhasingBackend,
    PhasingRequest,
    PipelineInterface,
    validate_real_inputs,
)
from agent.state import PhasingEvidence, RegionState
from pipeline.real_evidence import (
    REAL_EVIDENCE_DATA_STATUS,
    ExtractedRegionState,
    build_real_region_states,
)


DEFAULT_CONFIG_PATH = EXAMPLE_CONFIG_PATH


class RealSmokeError(RuntimeError):
    """Raised when the real-evidence smoke run is configured unsafely."""


class RealEvidenceAuditRecord(BaseModel):
    """One region audit that cannot be mistaken for a scored evaluation record."""

    model_config = ConfigDict(extra="forbid")

    audit_schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    recorded_at: datetime
    data_status: Literal["real_evidence_no_ground_truth"]
    scientific_interpretation_allowed: Literal[False] = False
    ground_truth_available: Literal[False] = False
    scoring_performed: Literal[False] = False
    execution_status: Literal["success", "error"]
    sample_id: str = Field(min_length=1)
    region_key: str = Field(min_length=1)
    model_provider: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    pipeline_mode: Literal["real"] = "real"
    duration_seconds: float = Field(ge=0.0)
    configuration: dict[str, object]
    initial_state: RegionState
    final_state: RegionState | None = None
    error_type: str | None = None
    error_message: str | None = None
    model_failure: dict[str, object] | None = None

    @model_validator(mode="after")
    def result_and_error_are_consistent(self) -> "RealEvidenceAuditRecord":
        if self.execution_status == "success":
            if self.final_state is None:
                raise ValueError("successful real-evidence audit requires final_state")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful real-evidence audit cannot contain an error")
        else:
            if self.final_state is not None:
                raise ValueError("failed real-evidence audit cannot contain final_state")
            if not self.error_type or not self.error_message:
                raise ValueError("failed real-evidence audit requires error details")
        return self


class EvidenceOnlyRealBackend:
    """Fallback when the recorded configuration exposes no retry strategies."""

    def phase(self, request: PhasingRequest) -> PhasingEvidence:
        raise RealSmokeError(
            "the real-evidence smoke test has no approved executable retry backend"
        )


def _audit_filename(index: int, region_key: str) -> str:
    digest = hashlib.sha256(region_key.encode("utf-8")).hexdigest()[:12]
    return f"region-{index:05d}-{digest}.json"


def write_real_audit(
    record: RealEvidenceAuditRecord,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one region audit without silently replacing a previous record."""

    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite real-evidence audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        record.model_dump_json(indent=2, exclude_computed_fields=True) + "\n",
        encoding="utf-8",
    )
    return path


def _record_from_existing(path: Path, expected_region_key: str) -> RealEvidenceAuditRecord:
    try:
        record = RealEvidenceAuditRecord.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise RealSmokeError(f"cannot resume from invalid audit file: {path}") from exc
    if record.region_key != expected_region_key:
        raise RealSmokeError(
            f"resume audit region mismatch in {path}: "
            f"{record.region_key} != {expected_region_key}"
        )
    return record


def render_smoke_summary(
    *,
    batch_id: str,
    sample_id: str,
    output_directory: Path,
    total_covered_regions: int,
    records: Sequence[RealEvidenceAuditRecord],
) -> str:
    """Render a short, explicitly non-evaluative progress/final summary."""

    initial_actions: Counter[str] = Counter()
    initial_sources: Counter[str] = Counter()
    terminal_actions: Counter[str] = Counter()
    rationales: list[tuple[str, str]] = []
    for record in records:
        if record.execution_status != "success" or record.final_state is None:
            continue
        if not record.final_state.decision_history:
            continue
        initial = record.final_state.decision_history[0]
        initial_actions[initial.action.value] += 1
        initial_sources[initial.source.value] += 1
        decision = record.final_state.decision
        if decision is not None:
            terminal_actions[decision.action.value] += 1
        if len(rationales) < 3:
            rationales.append((record.region_key, initial.rationale))

    errors = [record for record in records if record.execution_status == "error"]
    succeeded = len(records) - len(errors)
    lines = [
        "Haplotype Mapper real-evidence smoke test",
        f"Batch: {batch_id}",
        f"Sample: {sample_id}",
        f"Data status: {REAL_EVIDENCE_DATA_STATUS}",
        "Ground truth: unavailable",
        "Scoring performed: no",
        f"Output directory: {output_directory}",
        "",
        f"Covered target regions: {total_covered_regions}",
        f"Regions processed: {len(records)}",
        f"Regions remaining: {total_covered_regions - len(records)}",
        f"Successful graph runs: {succeeded}",
        f"Errors or crashes: {len(errors)}",
        "Initial decisions (all sources):",
        f"accept: {initial_actions['accept']}",
        f"retry: {initial_actions['retry']}",
        f"escalate: {initial_actions['escalate']}",
        f"deterministic: {initial_sources['deterministic']}",
        f"model: {initial_sources['llm']}",
        "Terminal graph outcomes:",
        f"accept: {terminal_actions['accept']}",
        f"escalate: {terminal_actions['escalate']}",
    ]
    if errors:
        lines.extend(["", "Example errors:"])
        for record in errors[:3]:
            lines.append(
                f"- {record.region_key}: {record.error_type}: {record.error_message}"
            )
    if rationales:
        lines.extend(["", "Example decision rationales:"])
        for region_key, rationale in rationales:
            lines.append(f"- {region_key}: {rationale}")
    lines.extend(
        [
            "",
            "Interpretation boundary: this is a plumbing/robustness run on real ",
            "biological evidence without sample-level labels. Decision counts are ",
            "descriptive only and must not be reported as accuracy or correctness.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_summary(
    output_directory: Path,
    *,
    batch_id: str,
    sample_id: str,
    total_covered_regions: int,
    records: Sequence[RealEvidenceAuditRecord],
) -> Path:
    path = output_directory / "summary.txt"
    path.write_text(
        render_smoke_summary(
            batch_id=batch_id,
            sample_id=sample_id,
            output_directory=output_directory,
            total_covered_regions=total_covered_regions,
            records=records,
        ),
        encoding="utf-8",
    )
    return path


def run_real_evidence_smoke(
    *,
    config: AgentConfig,
    extracted_states: Sequence[ExtractedRegionState],
    output_directory: str | Path,
    resume: bool = False,
    max_regions: int | None = None,
    progress_every: int | None = None,
    write_summary: bool = True,
    retry_inputs: Mapping[str, str | Path] | None = None,
    region_categories_path: str | Path = "data/ground_truth/derived/region_categories.csv",
) -> list[RealEvidenceAuditRecord]:
    """Run each covered real state through one shared compiled model graph."""

    if config.pipeline.mode != PipelineMode.REAL:
        raise RealSmokeError("real-evidence smoke runs require pipeline.mode=real")
    if config.pipeline.retry_strategies and not retry_inputs:
        raise RealSmokeError(
            "active retries require explicit original filtered VCFs via retry_inputs "
            "or --retry-input-vcf; do not reuse a phased VCF as the retry source"
        )
    if not extracted_states:
        raise RealSmokeError("no target regions with VCF variant coverage were found")
    if max_regions is not None and max_regions < 1:
        raise RealSmokeError("max_regions must be at least 1")
    progress_interval = (
        config.execution.real_smoke_progress_every
        if progress_every is None
        else progress_every
    )
    if progress_interval < 1:
        raise RealSmokeError("progress_every must be at least 1")

    selected_states = list(extracted_states[:max_regions])
    if config.pipeline.retry_strategies:
        samples = {item.state.sample_id for item in selected_states}
        if not samples.issubset(retry_inputs or {}):
            raise RealSmokeError("retry_inputs must cover every selected sample")
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(
            f"output directory is not empty; use --resume or choose another path: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    # Freeze actual Skills, not just their path, before any new inference.
    skills = load_skills(config.paths.skills_directory)
    plan = {
        "configuration": config.model_dump(mode="json"),
        "skills": {skill.name: {"content": skill.content, "sha256": hashlib.sha256(
            (config.paths.skills_directory / skill.name).read_bytes()).hexdigest()}
            for skill in skills},
        "retry_inputs": {sample: str(Path(path).resolve())
                         for sample, path in (retry_inputs or {}).items()},
        "region_categories": str(Path(region_categories_path).resolve()),
        "case_ids": [f"{item.state.sample_id}::{item.state.region.region_key}"
                     for item in selected_states],
        "case_state_sha256": [hashlib.sha256(item.state.model_dump_json(
            exclude_computed_fields=True,
            exclude={"attempt_history": {"__all__": {"recorded_at"}}},
        ).encode()).hexdigest() for item in selected_states],
    }
    plan_path = output / "assessment_plan.json"
    if plan_path.exists():
        if not resume or json.loads(plan_path.read_text()) != plan:
            raise RealSmokeError("assessment plan changed; use a new output directory")
    elif resume and any(output.iterdir()):
        raise RealSmokeError("existing run has no matching assessment plan; use a new output directory")
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")

    common_inputs = selected_states[0].state.inputs
    validate_real_inputs(
        common_inputs,
        expected_sample=selected_states[0].state.sample_id,
        expected_ploidy=config.pipeline.whatshap.ploidy,
    )
    backend: PhasingBackend = (
        WhatsHapRetryBackend(
            whatshap=config.pipeline.whatshap,
            strategies=config.pipeline.retry_strategies,
            input_vcfs=retry_inputs or {},
            region_categories_path=region_categories_path,
            output_directory=output / "retries",
            timeout_seconds=config.execution.whatshap_retry_timeout_seconds,
        ) if config.pipeline.retry_strategies else EvidenceOnlyRealBackend()
    )
    model = create_chat_model(config.model)
    assessor = ModelAssessor(model=model, skills=skills,
                            retry_strategies=config.pipeline.retry_strategies)
    graph = build_agent_graph(
        config=config,
        assessor=assessor,
        pipeline=PipelineInterface(
            backend,
            validate_inputs=True,
            expected_ploidy=config.pipeline.whatshap.ploidy,
        ),
    )
    recursion_limit = config.execution.graph_recursion_limit(
        config.decisions.max_retries
    )
    batch_id = output.name
    records: list[RealEvidenceAuditRecord] = []

    for index, extracted in enumerate(selected_states, start=1):
        state = extracted.state
        audit_path = output / _audit_filename(index, state.region.region_key)
        if audit_path.exists() and resume:
            record = _record_from_existing(audit_path, state.region.region_key)
            records.append(record)
            continue

        started = perf_counter()
        try:
            raw_final_state = graph.invoke(
                state.model_copy(deep=True),
                config={"recursion_limit": recursion_limit},
            )
            final_state = RegionState.model_validate(raw_final_state)
            record = RealEvidenceAuditRecord(
                run_id=str(uuid4()),
                batch_id=batch_id,
                recorded_at=datetime.now(timezone.utc),
                data_status=REAL_EVIDENCE_DATA_STATUS,
                execution_status="success",
                sample_id=state.sample_id,
                region_key=state.region.region_key,
                model_provider=config.model.provider,
                model_name=config.model.name,
                duration_seconds=perf_counter() - started,
                configuration=config.model_dump(mode="json"),
                initial_state=state,
                final_state=final_state,
            )
        except Exception as exc:
            record = RealEvidenceAuditRecord(
                run_id=str(uuid4()),
                batch_id=batch_id,
                recorded_at=datetime.now(timezone.utc),
                data_status=REAL_EVIDENCE_DATA_STATUS,
                execution_status="error",
                sample_id=state.sample_id,
                region_key=state.region.region_key,
                model_provider=config.model.provider,
                model_name=config.model.name,
                duration_seconds=perf_counter() - started,
                configuration=config.model_dump(mode="json"),
                initial_state=state,
                error_type=type(exc).__name__,
                error_message=str(exc),
                model_failure=(exc.audit_detail() if isinstance(exc, ModelAssessmentError) else None),
            )
        write_real_audit(record, audit_path)
        records.append(record)
        if write_summary:
            _write_summary(
                output,
                batch_id=batch_id,
                sample_id=state.sample_id,
                total_covered_regions=len(selected_states),
                records=records,
            )
        if (
            len(records) % progress_interval == 0
            or len(records) == len(selected_states)
        ):
            print(
                f"processed {len(records)}/{len(selected_states)} regions; "
                f"latest={record.execution_status}; region={record.region_key}",
                flush=True,
            )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a non-scoring, resumable Ollama smoke test on real phased VCF evidence."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--vcf", default="data/processed/test.phased.target_regions.vcf.gz"
    )
    parser.add_argument("--bam", default="data/processed/test.md.bam")
    parser.add_argument(
        "--bed", default="data/ground_truth/derived/target_regions.bed"
    )
    parser.add_argument(
        "--region-categories",
        default="data/ground_truth/derived/region_categories.csv",
    )
    parser.add_argument("--reference", default="data/reference/Egrandis_v2.fa")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-input-vcf", type=Path,
                        help="Original full filtered, unphased VCF for --sample; required for active retries.")
    parser.add_argument("--region-key", help="Run one exact target-region key from the extracted sample.")
    parser.add_argument("--retry-strategy-id", help="Restrict this run to one configured retry strategy ID.")
    parser.add_argument(
        "--max-regions",
        type=int,
        help="Optional development limit; omitted means every covered target region.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        help="Override execution.real_smoke_progress_every for this run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config).with_pipeline_mode(PipelineMode.REAL)
    if args.retry_strategy_id:
        selected = [item for item in config.pipeline.retry_strategies
                    if item.strategy_id == args.retry_strategy_id]
        if len(selected) != 1:
            raise RealSmokeError("--retry-strategy-id must name one configured strategy")
        config = config.model_copy(update={"pipeline": config.pipeline.model_copy(
            update={"retry_strategies": selected}
        )})
    extracted = build_real_region_states(
        vcf_path=args.vcf,
        bam_path=args.bam,
        bed_path=args.bed,
        region_categories_path=args.region_categories,
        reference_fasta=args.reference,
        expected_sample=args.sample,
    )
    if args.region_key:
        extracted = [item for item in extracted if item.state.region.region_key == args.region_key]
        if len(extracted) != 1:
            raise RealSmokeError("--region-key must identify exactly one covered target region")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output_dir or Path(
        f"outputs/real-evidence-smoke-{args.sample}-{timestamp}"
    )
    records = run_real_evidence_smoke(
        config=config,
        extracted_states=extracted,
        output_directory=output,
        resume=args.resume,
        max_regions=args.max_regions,
        progress_every=args.progress_every,
        retry_inputs=({args.sample: args.retry_input_vcf} if args.retry_input_vcf else None),
        region_categories_path=args.region_categories,
    )
    print((output / "summary.txt").read_text(encoding="utf-8"))
    return 0 if all(record.execution_status == "success" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EvidenceOnlyRealBackend",
    "RealEvidenceAuditRecord",
    "RealSmokeError",
    "render_smoke_summary",
    "run_real_evidence_smoke",
    "write_real_audit",
]

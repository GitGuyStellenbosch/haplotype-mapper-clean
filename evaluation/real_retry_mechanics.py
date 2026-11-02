"""Isolated, unvalidated proof that a real WhatsHap retry traverses the graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Sequence

from agent.config import AgentConfig, PipelineMode, WhatsHapConfig, load_config
from agent.graph import build_agent_graph
from agent.nodes import AssessmentResult
from agent.pipeline_interface import (
    PhasingRequest,
    PipelineError,
    PipelineInterface,
)
from agent.real_smoke import RealEvidenceAuditRecord
from agent.state import (
    DecisionAction,
    DecisionSource,
    InputFiles,
    PhasingEvidence,
    PhasingStrategy,
    RegionState,
)
from pipeline.real_evidence import (
    REAL_EVIDENCE_DATA_STATUS,
    build_real_region_states,
)
from pipeline.whatshap import PolyphaseCommandPlan, build_polyphase_argv


UNVALIDATED_STATUS = "unvalidated_mechanics_test_only"
EXECUTION_SCOPE = "sample_wide_cached_per_exact_strategy"
ARTIFACT_POLICY = "immutable_attempt_scoped_no_merge"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mechanics_strategy(path: str | Path) -> tuple[dict[str, object], PhasingStrategy]:
    """Load the one isolated strategy and reject anything resembling approval."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("validation_status") != UNVALIDATED_STATUS:
        raise ValueError(f"mechanics config must use {UNVALIDATED_STATUS!r}")
    if raw.get("scientifically_approved") is not False:
        raise ValueError("mechanics strategy must be scientifically unapproved")
    if raw.get("eligible_for_real_evaluation") is not False:
        raise ValueError("mechanics strategy must be excluded from real evaluation")
    if raw.get("execution_scope") != EXECUTION_SCOPE:
        raise ValueError(f"mechanics config must use execution scope {EXECUTION_SCOPE!r}")
    if raw.get("artifact_policy") != ARTIFACT_POLICY:
        raise ValueError(f"mechanics config must use artifact policy {ARTIFACT_POLICY!r}")
    strategy = PhasingStrategy.model_validate(raw.get("strategy"))
    if strategy.parameters.get("validation_status") != UNVALIDATED_STATUS:
        raise ValueError("strategy parameters must carry the unvalidated label")
    if strategy.parameters.get("scientifically_approved") is not False:
        raise ValueError("strategy parameters must record scientific approval as false")
    if set(strategy.parameters) != {
        "block_cut_sensitivity",
        "validation_status",
        "scientifically_approved",
    }:
        raise ValueError("mechanics strategy contains unsupported parameters")
    sensitivity = strategy.parameters["block_cut_sensitivity"]
    if not isinstance(sensitivity, int) or isinstance(sensitivity, bool):
        raise ValueError("block_cut_sensitivity must be an integer")
    return raw, strategy


class MechanicsAssessor:
    """Force one retry and then escalate; this is not a biological assessor."""

    def __init__(self, strategy: PhasingStrategy) -> None:
        self.strategy = strategy
        self.calls = 0

    def assess(self, state: RegionState) -> AssessmentResult:
        self.calls += 1
        if self.calls == 1:
            return AssessmentResult(
                action=DecisionAction.RETRY,
                confidence=1.0,
                rationale=(
                    "Unvalidated mechanics test: force exactly one real WhatsHap "
                    "retry; this is not a scientific decision."
                ),
                source=DecisionSource.DETERMINISTIC,
                retry_strategy=self.strategy,
            )
        return AssessmentResult(
            action=DecisionAction.ESCALATE,
            confidence=1.0,
            rationale=(
                "Unvalidated mechanics test complete after reassessment; escalate "
                "without interpreting whether the retry improved the biology."
            ),
            source=DecisionSource.DETERMINISTIC,
        )


class IsolatedWhatsHapMechanicsBackend:
    """Execute only the one unvalidated strategy through the existing argv builder."""

    def __init__(
        self,
        *,
        whatshap: WhatsHapConfig,
        strategy: PhasingStrategy,
        output_vcf: str | Path,
        region_categories_path: str | Path,
    ) -> None:
        self.whatshap = whatshap
        self.strategy = strategy
        self.output_vcf = Path(output_vcf)
        self.region_categories_path = Path(region_categories_path)
        self.executed_argv: list[str] | None = None
        self.duration_seconds: float | None = None
        self.stdout = ""
        self.stderr = ""
        self.execution_count = 0
        self.output_sha256: str | None = None
        self._request_signature: tuple[object, ...] | None = None
        self._evidence_by_region: dict[str, PhasingEvidence] = {}

    def _signature(self, request: PhasingRequest) -> tuple[object, ...]:
        return (
            request.sample_id,
            request.inputs.bam,
            request.inputs.vcf,
            request.inputs.reference_fasta,
            request.inputs.target_regions_bed,
            request.strategy.strategy_id,
            json.dumps(request.strategy.parameters, sort_keys=True),
        )

    def _normalize_evidence(
        self, evidence: PhasingEvidence, sensitivity: int
    ) -> PhasingEvidence:
        messages = [
            message
            for message in evidence.tool_messages
            if not message.startswith("Evidence extracted from an existing WhatsHap")
        ]
        messages.insert(
            0,
            "Evidence extracted from an isolated, unvalidated mechanics-test "
            "WhatsHap retry; no biological interpretation is permitted.",
        )
        return evidence.model_copy(
            update={
                "tool_messages": messages,
                "additional_metrics": {
                    **evidence.additional_metrics,
                    "mechanics_retry_validation_status": UNVALIDATED_STATUS,
                    "mechanics_retry_execution_scope": EXECUTION_SCOPE,
                    "mechanics_retry_artifact_policy": ARTIFACT_POLICY,
                    "whatshap_block_cut_sensitivity": sensitivity,
                },
            }
        )

    def phase(self, request: PhasingRequest) -> PhasingEvidence:
        if request.strategy != self.strategy:
            raise PipelineError("isolated backend refuses any other retry strategy")
        if request.inputs.vcf is None or request.inputs.bam is None:
            raise PipelineError("real retry requires VCF and BAM inputs")
        if request.inputs.reference_fasta is None:
            raise PipelineError("real retry requires a reference FASTA")

        sensitivity = int(request.strategy.parameters["block_cut_sensitivity"])
        signature = self._signature(request)
        if self._evidence_by_region:
            if signature != self._request_signature:
                raise PipelineError(
                    "cached sample-wide retry cannot be reused for different inputs or strategy"
                )
            try:
                artifact_changed = (
                    self.output_sha256 is None
                    or _sha256(self.output_vcf) != self.output_sha256
                )
            except OSError as exc:
                raise PipelineError("cached retry artifact is unavailable") from exc
            if artifact_changed:
                raise PipelineError("cached retry artifact changed after creation")
            try:
                return self._evidence_by_region[request.region.region_key]
            except KeyError as exc:
                raise PipelineError(
                    "cached retry output has no matching target region"
                ) from exc

        if self.output_vcf.exists():
            raise PipelineError(f"refusing to overwrite retry output: {self.output_vcf}")
        plan = PolyphaseCommandPlan.model_validate(
            {
                **self.whatshap.model_dump(),
                "block_cut_sensitivity": sensitivity,
                "input_vcf": request.inputs.vcf,
                "phase_inputs": [request.inputs.bam],
                "output_vcf": self.output_vcf,
                "reference_fasta": request.inputs.reference_fasta,
                "samples": [request.sample_id],
            }
        )
        self.executed_argv = build_polyphase_argv(plan)
        started = perf_counter()
        completed = subprocess.run(
            self.executed_argv,
            capture_output=True,
            text=True,
            check=False,
        )
        self.duration_seconds = perf_counter() - started
        self.stdout = completed.stdout
        self.stderr = completed.stderr
        if completed.returncode != 0:
            raise PipelineError(
                f"WhatsHap mechanics retry failed with exit code {completed.returncode}"
            )
        if not self.output_vcf.is_file():
            raise PipelineError("WhatsHap reported success without creating its output")
        self.execution_count += 1
        self.output_sha256 = _sha256(self.output_vcf)
        self._request_signature = signature

        extracted = build_real_region_states(
            vcf_path=self.output_vcf,
            bam_path=request.inputs.bam,
            bed_path=request.inputs.target_regions_bed,
            region_categories_path=self.region_categories_path,
            reference_fasta=request.inputs.reference_fasta,
            expected_sample=request.sample_id,
        )
        self._evidence_by_region = {
            item.state.region.region_key: self._normalize_evidence(
                item.state.evidence, sensitivity
            )
            for item in extracted
        }
        if len(self._evidence_by_region) != len(extracted):
            raise PipelineError("retry output contains duplicate target-region keys")
        try:
            return self._evidence_by_region[request.region.region_key]
        except KeyError as exc:
            raise PipelineError(
                "retry output did not contain exactly one matching target region"
            ) from exc


def _mechanics_runtime_config(
    production_config: AgentConfig, strategy: PhasingStrategy
) -> AgentConfig:
    if production_config.pipeline.retry_strategies:
        raise ValueError("production retry_strategies must remain empty")
    pipeline = production_config.pipeline.model_copy(
        update={"mode": PipelineMode.REAL, "retry_strategies": [strategy]}
    )
    return production_config.model_copy(update={"pipeline": pipeline})


def run_mechanics_proof(
    *,
    production_config: AgentConfig,
    mechanics_config: dict[str, object],
    strategy: PhasingStrategy,
    source_audit: RealEvidenceAuditRecord,
    retry_input_vcf: str | Path,
    region_categories_path: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    """Run one forced real retry and return a self-describing mechanics audit."""

    if source_audit.execution_status != "success" or source_audit.final_state is None:
        raise ValueError("source Phase-12 audit must be successful")
    if source_audit.final_state.decision is None:
        raise ValueError("source Phase-12 audit has no terminal decision")
    if source_audit.final_state.decision.action != DecisionAction.ESCALATE:
        raise ValueError("source Phase-12 case must previously have escalated")
    if source_audit.initial_state.metadata.get("data_status") != REAL_EVIDENCE_DATA_STATUS:
        raise ValueError("source case must be real evidence without ground truth")

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty mechanics output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    initial = source_audit.initial_state.model_copy(deep=True)
    initial.inputs = InputFiles(
        bam=initial.inputs.bam,
        vcf=Path(retry_input_vcf),
        reference_fasta=initial.inputs.reference_fasta,
        target_regions_bed=initial.inputs.target_regions_bed,
    )
    initial.metadata.update(
        {
            "mechanics_test_status": UNVALIDATED_STATUS,
            "scientifically_approved": False,
            "eligible_for_real_evaluation": False,
        }
    )
    runtime_config = _mechanics_runtime_config(production_config, strategy)
    retry_vcf = output / (
        f"{initial.sample_id}.unvalidated-mechanics-bcs"
        f"{strategy.parameters['block_cut_sensitivity']}.vcf.gz"
    )
    backend = IsolatedWhatsHapMechanicsBackend(
        whatshap=runtime_config.pipeline.whatshap,
        strategy=strategy,
        output_vcf=retry_vcf,
        region_categories_path=region_categories_path,
    )
    assessor = MechanicsAssessor(strategy)
    graph = build_agent_graph(
        config=runtime_config,
        assessor=assessor,
        pipeline=PipelineInterface(
            backend,
            validate_inputs=True,
            expected_ploidy=runtime_config.pipeline.whatshap.ploidy,
        ),
    )
    raw_final = graph.invoke(
        initial,
        config={
            "recursion_limit": runtime_config.execution.graph_recursion_limit(
                runtime_config.decisions.max_retries
            )
        },
    )
    final = RegionState.model_validate(raw_final)
    if len(final.attempt_history) != len(initial.attempt_history) + 1:
        raise RuntimeError("mechanics proof did not record exactly one new attempt")
    if final.attempt_history[-1].strategy_id != strategy.strategy_id:
        raise RuntimeError("mechanics proof recorded the wrong retry strategy")
    if assessor.calls != 2:
        raise RuntimeError("mechanics proof did not reassess exactly once")
    if final.decision is None or final.decision.action not in {
        DecisionAction.ACCEPT,
        DecisionAction.ESCALATE,
    }:
        raise RuntimeError("mechanics proof did not terminate")

    report = {
        "report_schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": mechanics_config["purpose"],
        "validation_status": UNVALIDATED_STATUS,
        "scientifically_approved": False,
        "eligible_for_real_evaluation": False,
        "data_status": REAL_EVIDENCE_DATA_STATUS,
        "action_ground_truth_available": False,
        "scoring_performed": False,
        "source_phase12_run_id": source_audit.run_id,
        "source_phase12_case_id": (
            f"{source_audit.sample_id}::{source_audit.region_key}"
        ),
        "source_phase12_terminal_action": "escalate",
        "strategy": strategy.model_dump(mode="json"),
        "execution_scope": mechanics_config["execution_scope"],
        "artifact_policy": mechanics_config["artifact_policy"],
        "retry_artifact_sha256": backend.output_sha256,
        "whatshap_execution_count": backend.execution_count,
        "production_retry_strategies": production_config.pipeline.retry_strategies,
        "command_argv": backend.executed_argv,
        "whatshap_duration_seconds": backend.duration_seconds,
        "assessor_kind": "forced_mechanics_test_not_biological_assessment",
        "assessor_calls": assessor.calls,
        "retry_triggered": True,
        "new_attempt_recorded": True,
        "reassessed_after_retry": True,
        "terminal_action": final.decision.action.value,
        "initial_state": initial.model_dump(mode="json", exclude_computed_fields=True),
        "final_state": final.model_dump(mode="json", exclude_computed_fields=True),
        "interpretation_boundary": (
            "This proves graph-to-WhatsHap retry mechanics only. The strategy is "
            "unvalidated, scientifically unapproved, and forbidden from real evaluation."
        ),
    }
    (output / "audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / "whatshap.stdout.log").write_text(backend.stdout, encoding="utf-8")
    (output / "whatshap.stderr.log").write_text(backend.stderr, encoding="utf-8")
    (output / "summary.txt").write_text(
        "\n".join(
            (
                "Isolated real WhatsHap retry mechanics proof",
                f"Validation status: {UNVALIDATED_STATUS}",
                "Scientifically approved: no",
                "Eligible for real evaluation: no",
                f"Sample: {initial.sample_id}",
                f"Region: {initial.region.region_key}",
                f"Retry strategy: {strategy.strategy_id}",
                "Retry triggered: yes",
                "New attempt recorded: yes",
                f"Attempts: {len(initial.attempt_history)} -> {len(final.attempt_history)}",
                f"Assessments: {assessor.calls}",
                f"Terminal action: {final.decision.action.value}",
                f"WhatsHap seconds: {backend.duration_seconds:.3f}",
                "Interpretation: mechanics only; no strategy approval or accuracy claim.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mechanics-config", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--retry-input-vcf", type=Path, required=True)
    parser.add_argument("--region-categories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    mechanics_config, strategy = load_mechanics_strategy(args.mechanics_config)
    source_audit = RealEvidenceAuditRecord.model_validate_json(
        args.source_audit.read_text(encoding="utf-8")
    )
    report = run_mechanics_proof(
        production_config=load_config(args.config),
        mechanics_config=mechanics_config,
        strategy=strategy,
        source_audit=source_audit,
        retry_input_vcf=args.retry_input_vcf,
        region_categories_path=args.region_categories,
        output_directory=args.output_dir,
    )
    print((args.output_dir / "summary.txt").read_text(encoding="utf-8"))
    return 0 if report["terminal_action"] in {"accept", "escalate"} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_POLICY",
    "EXECUTION_SCOPE",
    "IsolatedWhatsHapMechanicsBackend",
    "MechanicsAssessor",
    "UNVALIDATED_STATUS",
    "load_mechanics_strategy",
    "run_mechanics_proof",
]

"""Execute allowlisted research retries using the established sample-wide scope."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

from agent.config import WhatsHapConfig
from agent.pipeline_interface import PhasingRequest, PipelineError, validate_real_inputs
from agent.state import InputFiles, PhasingEvidence, PhasingStrategy
from pipeline.real_evidence import build_real_region_states, read_single_sample_snp_vcf
from pipeline.whatshap import PolyphaseCommandPlan, build_polyphase_argv


RETRY_POLICY_VERSION = "research-retries-20261006-v2"
EXECUTION_SCOPE = "sample_wide_cached_per_exact_strategy"
ARTIFACT_POLICY = "immutable_attempt_scoped_no_merge"
SUPPORTED_STRATEGIES = {
    "lower-block-cut-sensitivity-v1": {"block_cut_sensitivity": 2},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _genotypes(path: Path, sample: str, *, require_unphased: bool = False) -> dict:
    """Keep variant identity and duplicate multiplicity, ignoring phase order."""
    _, chromosomes = read_single_sample_snp_vcf(path, expected_sample=sample)
    result = defaultdict(list)
    for chromosome, observations in chromosomes.items():
        for observation in observations:
            if require_unphased and observation.phased:
                raise PipelineError("retry source must be the original unphased filtered VCF")
            key = (chromosome, observation.position,
                   observation.reference_allele, observation.alternate_allele)
            result[key].append(tuple(sorted(re.split(r"[/|]", observation.genotype))))
    return {key: tuple(sorted(values)) for key, values in result.items()}


class WhatsHapRetryBackend:
    """One immutable artifact per sample/strategy, cached for regional requests.

    The original filtered, unphased VCF must be supplied explicitly. Retries are
    independent alternatives to that input; no output is merged into baseline.
    """

    def __init__(
        self,
        *,
        whatshap: WhatsHapConfig,
        strategies: Sequence[PhasingStrategy],
        input_vcfs: Mapping[str, str | Path],
        region_categories_path: str | Path,
        output_directory: str | Path,
        timeout_seconds: int,
    ) -> None:
        self.whatshap = whatshap.model_copy(deep=True)
        if self.whatshap.distrust_genotypes:
            raise ValueError("phasing-only retries require trusted input genotypes")
        self.strategies = {s.strategy_id: s.model_copy(deep=True) for s in strategies}
        if len(self.strategies) != len(strategies):
            raise ValueError("retry strategy IDs must be unique")
        for strategy in self.strategies.values():
            expected = SUPPORTED_STRATEGIES.get(strategy.strategy_id)
            # Dict equality alone equates True with 1 and 2 with 2.0.
            if expected is None or strategy.parameters != expected or any(
                type(strategy.parameters[k]) is not type(v) for k, v in expected.items()
            ):
                raise ValueError(f"unsupported retry strategy: {strategy.strategy_id}")
        if timeout_seconds <= 0:
            raise ValueError("retry timeout must be positive")
        self.timeout_seconds = timeout_seconds
        self.input_vcfs = {sample: Path(p).resolve() for sample, p in input_vcfs.items()}
        self.region_categories_path = Path(region_categories_path).resolve()
        self.output_directory = Path(output_directory).resolve()
        self._cache: dict[tuple[str, str], tuple[tuple, Path, str, dict]] = {}
        self._failures: dict[tuple[str, str], str] = {}

    def phase(self, request: PhasingRequest) -> PhasingEvidence:
        strategy = self.strategies.get(request.strategy.strategy_id)
        if strategy is None or request.strategy != strategy or any(
            type(request.strategy.parameters[k]) is not type(v)
            for k, v in strategy.parameters.items()
        ):
            raise PipelineError("backend refuses unlisted or altered retry strategy")
        key = (request.sample_id, strategy.strategy_id)
        if key in self._failures:
            raise PipelineError(self._failures[key])
        try:
            input_vcf = self.input_vcfs[request.sample_id]
        except KeyError as exc:
            raise PipelineError("no original filtered retry VCF for this sample") from exc
        paths = [input_vcf, request.inputs.bam, request.inputs.reference_fasta,
                 request.inputs.target_regions_bed, self.region_categories_path]
        if any(p is None for p in paths):
            raise PipelineError("retry requires VCF, BAM, reference, BED, and catalogue")
        try:
            signature = tuple(
                (str(Path(p).resolve()), Path(p).stat().st_size, Path(p).stat().st_mtime_ns)
                for p in paths
            ) + (_sha256(input_vcf),)
        except OSError as exc:
            raise PipelineError("retry input is unavailable") from exc

        if key in self._cache:
            prior, output, digest, by_region = self._cache[key]
            if signature != prior:
                raise PipelineError("retry inputs changed after the cached attempt")
            if not output.is_file() or _sha256(output) != digest:
                raise PipelineError("cached retry artifact changed or disappeared")
            return self._regional(by_region, request)

        validate_real_inputs(
            InputFiles(**{**request.inputs.model_dump(), "vcf": input_vcf}),
            expected_sample=request.sample_id,
            expected_ploidy=self.whatshap.ploidy,
        )
        before = _genotypes(input_vcf, request.sample_id, require_unphased=True)
        token = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
        directory = self.output_directory / token
        output = directory / "phased.vcf.gz"
        plan = PolyphaseCommandPlan.model_validate({
            **self.whatshap.model_dump(), **strategy.parameters,
            "input_vcf": input_vcf, "phase_inputs": [request.inputs.bam],
            "output_vcf": output, "reference_fasta": request.inputs.reference_fasta,
            "samples": [request.sample_id],
        })
        argv = build_polyphase_argv(plan)
        if directory.exists():
            try:
                saved = json.loads((directory / "attempt.json").read_text())
            except (OSError, ValueError) as exc:
                raise PipelineError("refusing existing incomplete retry directory") from exc
            if (
                saved.get("status") != "success"
                or saved.get("command_argv") != argv
                or saved.get("input_signature") != json.loads(json.dumps(signature))
                or not output.is_file()
                or _sha256(output) != saved.get("output_sha256")
            ):
                raise PipelineError("refusing changed or failed saved retry attempt")
            changes, by_region, digest = self._extract(output, before, request, strategy)
            self._cache[key] = (signature, output, digest, by_region)
            return self._regional(by_region, request)
        directory.mkdir(parents=True, exist_ok=False)
        audit = {
            "retry_policy_version": RETRY_POLICY_VERSION,
            "scientifically_validated": False,
            "execution_scope": EXECUTION_SCOPE,
            "artifact_policy": ARTIFACT_POLICY,
            "sample_id": request.sample_id,
            "trigger_region_key": request.region.region_key,
            "strategy": strategy.model_dump(mode="json"),
            "input_vcf": str(input_vcf), "input_sha256": signature[-1],
            "input_signature": signature, "output_vcf": str(output),
            "command_argv": argv, "timeout_seconds": self.timeout_seconds,
            "status": "running",
        }
        (directory / "attempt.json").write_text(json.dumps(audit, indent=2) + "\n")
        started = perf_counter()
        try:
            completed = subprocess.run(argv, capture_output=True, text=True,
                                       check=False, timeout=self.timeout_seconds)
            (directory / "stdout.log").write_text(completed.stdout)
            (directory / "stderr.log").write_text(completed.stderr)
            if completed.returncode != 0:
                raise PipelineError(f"WhatsHap retry failed with exit code {completed.returncode}")
            if not output.is_file():
                raise PipelineError("WhatsHap reported success without an output VCF")
            changes, by_region, digest = self._extract(output, before, request, strategy)
            result = self._regional(by_region, request)
            audit.update(status="success", output_sha256=digest,
                         genotype_change_count=len(changes))
            self._cache[key] = (signature, output, digest, by_region)
            return result
        except Exception as exc:
            audit.update(status="error", error_type=type(exc).__name__, error_message=str(exc))
            message = f"research retry stopped: {type(exc).__name__}: {exc}"
            self._failures[key] = message
            if isinstance(exc, subprocess.TimeoutExpired):
                (directory / "stderr.log").write_text(str(exc))
            raise PipelineError(message) from exc
        finally:
            audit["duration_seconds"] = perf_counter() - started
            (directory / "attempt.json").write_text(json.dumps(audit, indent=2) + "\n")

    def _extract(self, output: Path, before: dict, request: PhasingRequest,
                 strategy: PhasingStrategy) -> tuple[list, dict, str]:
        after = _genotypes(output, request.sample_id)
        if before.keys() != after.keys() or any(
            len(before[key]) != len(after[key]) for key in before
        ):
            raise PipelineError("retry changed the retained SNP records or alleles")
        # Duplicate variants are present in the real filtered input. Compare
        # genotype multisets within each identity, without assuming row order.
        changes = [key for key in before for _ in range(sum(
            (Counter(before[key]) - Counter(after[key])).values()
        ))]
        if changes:
            raise PipelineError("phasing-only retry changed allele dosage")
        extracted = build_real_region_states(
            vcf_path=output, bam_path=request.inputs.bam,
            bed_path=request.inputs.target_regions_bed,
            region_categories_path=self.region_categories_path,
            reference_fasta=request.inputs.reference_fasta,
            expected_sample=request.sample_id,
        )
        digest = _sha256(output)
        input_digest = _sha256(self.input_vcfs[request.sample_id])
        by_region = {}
        for item in extracted:
            region = item.state.region
            evidence = item.state.evidence.model_copy(deep=True)
            evidence.tool_messages = [message for message in evidence.tool_messages
                if not message.startswith("Evidence extracted from an existing WhatsHap")]
            evidence.tool_messages.insert(0, "Evidence from an active research retry; biological correctness is unvalidated.")
            evidence.additional_metrics.update({
                "retry_policy_version": RETRY_POLICY_VERSION,
                "retry_strategy_id": strategy.strategy_id,
                "retry_execution_scope": EXECUTION_SCOPE,
                "retry_output_vcf": str(output.resolve()),
                "retry_output_sha256": digest,
                "retry_input_vcf": str(self.input_vcfs[request.sample_id]),
                "retry_input_sha256": input_digest,
                "retry_genotype_change_count": 0,
            })
            if region.region_key in by_region:
                raise PipelineError("retry produced duplicate region keys")
            by_region[region.region_key] = evidence
        return changes, by_region, digest

    @staticmethod
    def _regional(by_region: dict, request: PhasingRequest) -> PhasingEvidence:
        try:
            return by_region[request.region.region_key].model_copy(deep=True)
        except KeyError as exc:
            raise PipelineError("retry output has no matching target region") from exc

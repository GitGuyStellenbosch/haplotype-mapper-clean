"""Validated, non-executing command plans for WhatsHap polyphase 2.8."""

from __future__ import annotations

import shlex
from pathlib import Path

from pydantic import Field, model_validator

from agent.config import WhatsHapConfig


class WhatsHapCommandError(ValueError):
    """Raised when a requested dry-run command would be invalid or unsafe."""


class PolyphaseCommandPlan(WhatsHapConfig):
    """Complete declarative plan for one `whatshap polyphase` invocation.

    This model builds arguments only. It does not inspect biological contents,
    create directories, or execute WhatsHap.
    """

    input_vcf: Path
    phase_inputs: list[Path] = Field(min_length=1)
    output_vcf: Path
    reference_fasta: Path | None = None
    samples: list[str] = Field(default_factory=list)
    chromosomes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def plan_is_consistent(self) -> "PolyphaseCommandPlan":
        if self.use_reference and self.reference_fasta is None:
            raise ValueError("reference_fasta is required when use_reference=True")
        if self.input_vcf == self.output_vcf:
            raise ValueError("input_vcf and output_vcf must differ")
        if len(self.phase_inputs) != len(set(self.phase_inputs)):
            raise ValueError("phase_inputs must be unique")
        if len(self.samples) != len(set(self.samples)):
            raise ValueError("samples must be unique")
        if len(self.chromosomes) != len(set(self.chromosomes)):
            raise ValueError("chromosomes must be unique")
        return self


def build_polyphase_argv(plan: PolyphaseCommandPlan) -> list[str]:
    """Return an argument vector suitable for a future subprocess call."""

    argv = [
        plan.executable,
        "polyphase",
        "--output",
        str(plan.output_vcf),
        "--ploidy",
        str(plan.ploidy),
        "--block-cut-sensitivity",
        str(plan.block_cut_sensitivity),
        "--tag",
        plan.tag,
        "--mapping-quality",
        str(plan.mapping_quality),
        "--threads",
        str(plan.threads),
        "--min-overlap",
        str(plan.min_overlap),
    ]

    if plan.use_reference:
        argv.extend(["--reference", str(plan.reference_fasta)])
    for sample in plan.samples:
        argv.extend(["--sample", sample])
    for chromosome in plan.chromosomes:
        argv.extend(["--chromosome", chromosome])

    enabled_flags = (
        (plan.only_snvs, "--only-snvs"),
        (plan.ignore_read_groups, "--ignore-read-groups"),
        (plan.include_haploid_sets, "--include-haploid-sets"),
        (plan.distrust_genotypes, "--distrust-genotypes"),
        (plan.use_prephasing, "--use-prephasing"),
        (plan.no_mav, "--no-mav"),
        (plan.use_supplementary, "--use-supplementary"),
    )
    argv.extend(flag for enabled, flag in enabled_flags if enabled)
    if plan.use_supplementary:
        argv.extend(
            ["--supplementary-distance", str(plan.supplementary_distance)]
        )

    argv.append(str(plan.input_vcf))
    argv.extend(str(path) for path in plan.phase_inputs)
    return argv


def render_polyphase_command(plan: PolyphaseCommandPlan) -> str:
    """Render the argv safely for human review; this string is never executed."""

    return shlex.join(build_polyphase_argv(plan))


__all__ = [
    "PolyphaseCommandPlan",
    "WhatsHapCommandError",
    "build_polyphase_argv",
    "render_polyphase_command",
]

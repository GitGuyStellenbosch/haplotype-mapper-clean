"""Validated, non-executing plans for GATK calling and bcftools selection."""

from __future__ import annotations

import shlex
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.config import (
    AgentConfig,
    GATKGenotypeGVCFsConfig,
    GATKHardFilterConfig,
    GATKHaplotypeCallerConfig,
    VariantSelectionConfig,
)


class HaplotypeCallerCommandPlan(GATKHaplotypeCallerConfig):
    """Declarative plan for one GATK HaplotypeCaller GVCF invocation."""

    executable: str = Field(min_length=1)
    input_bam: Path
    reference_fasta: Path
    output_gvcf: Path

    @model_validator(mode="after")
    def input_and_output_differ(self) -> "HaplotypeCallerCommandPlan":
        if self.input_bam == self.output_gvcf:
            raise ValueError("HaplotypeCaller input and output must differ")
        return self


class GenotypeGVCFsCommandPlan(GATKGenotypeGVCFsConfig):
    """Declarative plan for GATK GenotypeGVCFs.

    A null sample_ploidy deliberately omits the argument. It does not assert that
    diploid or tetraploid behavior is scientifically correct.
    """

    executable: str = Field(min_length=1)
    input_gvcf: Path
    reference_fasta: Path
    output_vcf: Path

    @model_validator(mode="after")
    def input_and_output_differ(self) -> "GenotypeGVCFsCommandPlan":
        if self.input_gvcf == self.output_vcf:
            raise ValueError("GenotypeGVCFs input and output must differ")
        return self


class VariantFiltrationCommandPlan(GATKHardFilterConfig):
    """Declarative plan for the seven recorded GATK hard filters."""

    executable: str = Field(min_length=1)
    input_vcf: Path
    reference_fasta: Path
    output_vcf: Path

    @model_validator(mode="after")
    def input_and_output_differ(self) -> "VariantFiltrationCommandPlan":
        if self.input_vcf == self.output_vcf:
            raise ValueError("VariantFiltration input and output must differ")
        return self


class VariantSelectionCommandPlan(VariantSelectionConfig):
    """Declarative plan for bcftools PASS/biallelic/SNP selection."""

    input_vcf: Path
    output_vcf: Path

    @model_validator(mode="after")
    def input_and_output_differ(self) -> "VariantSelectionCommandPlan":
        if self.input_vcf == self.output_vcf:
            raise ValueError("bcftools selection input and output must differ")
        return self


class VariantCallingPlans(BaseModel):
    """Complete upstream command sequence; data only, with no execution method."""

    model_config = ConfigDict(extra="forbid")

    haplotype_caller: HaplotypeCallerCommandPlan
    genotype_gvcfs: GenotypeGVCFsCommandPlan
    variant_filtration: VariantFiltrationCommandPlan
    variant_selection: VariantSelectionCommandPlan

    @model_validator(mode="after")
    def stages_form_one_chain(self) -> "VariantCallingPlans":
        if self.haplotype_caller.output_gvcf != self.genotype_gvcfs.input_gvcf:
            raise ValueError("HaplotypeCaller output must feed GenotypeGVCFs")
        if self.genotype_gvcfs.output_vcf != self.variant_filtration.input_vcf:
            raise ValueError("GenotypeGVCFs output must feed VariantFiltration")
        if self.variant_filtration.output_vcf != self.variant_selection.input_vcf:
            raise ValueError("VariantFiltration output must feed bcftools selection")
        references = {
            self.haplotype_caller.reference_fasta,
            self.genotype_gvcfs.reference_fasta,
            self.variant_filtration.reference_fasta,
        }
        if len(references) != 1:
            raise ValueError("all GATK stages must use the same reference")
        return self


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def build_haplotype_caller_argv(plan: HaplotypeCallerCommandPlan) -> list[str]:
    """Build argv only; never invoke GATK."""

    return [
        plan.executable,
        "HaplotypeCaller",
        "--sample-ploidy",
        str(plan.sample_ploidy),
        "--emit-ref-confidence",
        plan.emit_reference_confidence,
        "--output",
        str(plan.output_gvcf),
        "--input",
        str(plan.input_bam),
        "--reference",
        str(plan.reference_fasta),
    ]


def build_genotype_gvcfs_argv(plan: GenotypeGVCFsCommandPlan) -> list[str]:
    """Build argv while keeping unresolved sample ploidy optional."""

    argv = [
        plan.executable,
        "GenotypeGVCFs",
        "--output",
        str(plan.output_vcf),
        "--variant",
        str(plan.input_gvcf),
        "--reference",
        str(plan.reference_fasta),
        "--include-non-variant-sites",
        _bool_text(plan.include_non_variant_sites),
    ]
    if plan.sample_ploidy is not None:
        argv.extend(["--sample-ploidy", str(plan.sample_ploidy)])
    return argv


def _filter_specs(plan: VariantFiltrationCommandPlan) -> list[tuple[str, str]]:
    values = (
        ("QD", "<", plan.minimum_qd),
        ("QUAL", "<", plan.minimum_qual),
        ("SOR", ">", plan.maximum_sor),
        ("FS", ">", plan.maximum_fs),
        ("MQ", "<", plan.minimum_mq),
        ("MQRankSum", "<", plan.minimum_mq_rank_sum),
        ("ReadPosRankSum", "<", plan.minimum_read_pos_rank_sum),
    )
    return [
        (f"{field} {operator} {value}", f"{field}{value:g}")
        for field, operator, value in values
    ]


def build_variant_filtration_argv(plan: VariantFiltrationCommandPlan) -> list[str]:
    """Build argv for the recorded seven hard-filter expressions."""

    filters = _filter_specs(plan)
    argv = [
        plan.executable,
        "VariantFiltration",
        "--output",
        str(plan.output_vcf),
    ]
    for expression, _ in filters:
        argv.extend(["--filter-expression", expression])
    for _, name in filters:
        argv.extend(["--filter-name", name])
    argv.extend(
        [
            "--variant",
            str(plan.input_vcf),
            "--reference",
            str(plan.reference_fasta),
        ]
    )
    return argv


def build_variant_selection_argv(plan: VariantSelectionCommandPlan) -> list[str]:
    """Build argv matching the recorded bcftools view selection step."""

    argv = [plan.executable, "view"]
    if plan.require_pass:
        argv.extend(["-f", "PASS"])
    argv.extend([f"-m{plan.minimum_alleles}", f"-M{plan.maximum_alleles}"])
    if plan.snps_only:
        argv.extend(["-v", "snps"])
    argv.extend(["-Oz", "-o", str(plan.output_vcf), str(plan.input_vcf)])
    return argv


def render_command(argv: list[str]) -> str:
    """Safely render an argv for review; the resulting string is not executed."""

    return shlex.join(argv)


def build_variant_calling_plans(
    *,
    config: AgentConfig,
    input_bam: str | Path,
    reference_fasta: str | Path,
    output_gvcf: str | Path,
    output_vcf: str | Path,
    filtered_vcf: str | Path,
    selected_vcf: str | Path,
) -> VariantCallingPlans:
    """Construct the four-stage dry-run chain from canonical configuration."""

    gatk = config.pipeline.gatk
    reference = Path(reference_fasta)
    haplotype_caller = HaplotypeCallerCommandPlan.model_validate(
        {
            **gatk.haplotype_caller.model_dump(),
            "executable": gatk.executable,
            "input_bam": Path(input_bam),
            "reference_fasta": reference,
            "output_gvcf": Path(output_gvcf),
        }
    )
    genotype_gvcfs = GenotypeGVCFsCommandPlan.model_validate(
        {
            **gatk.genotype_gvcfs.model_dump(),
            "executable": gatk.executable,
            "input_gvcf": Path(output_gvcf),
            "reference_fasta": reference,
            "output_vcf": Path(output_vcf),
        }
    )
    variant_filtration = VariantFiltrationCommandPlan.model_validate(
        {
            **gatk.hard_filters.model_dump(),
            "executable": gatk.executable,
            "input_vcf": Path(output_vcf),
            "reference_fasta": reference,
            "output_vcf": Path(filtered_vcf),
        }
    )
    variant_selection = VariantSelectionCommandPlan.model_validate(
        {
            **gatk.post_filter_selection.model_dump(),
            "input_vcf": Path(filtered_vcf),
            "output_vcf": Path(selected_vcf),
        }
    )
    return VariantCallingPlans(
        haplotype_caller=haplotype_caller,
        genotype_gvcfs=genotype_gvcfs,
        variant_filtration=variant_filtration,
        variant_selection=variant_selection,
    )


__all__ = [
    "GenotypeGVCFsCommandPlan",
    "HaplotypeCallerCommandPlan",
    "VariantCallingPlans",
    "VariantFiltrationCommandPlan",
    "VariantSelectionCommandPlan",
    "build_genotype_gvcfs_argv",
    "build_haplotype_caller_argv",
    "build_variant_calling_plans",
    "build_variant_filtration_argv",
    "build_variant_selection_argv",
    "render_command",
]

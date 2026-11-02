"""Tests for non-executing GATK and bcftools command plans."""

from pathlib import Path
from unittest import TestCase

from pydantic import ValidationError

from agent.config import load_config
from pipeline.variant_calling import (
    GenotypeGVCFsCommandPlan,
    VariantCallingPlans,
    build_genotype_gvcfs_argv,
    build_haplotype_caller_argv,
    build_variant_calling_plans,
    build_variant_filtration_argv,
    build_variant_selection_argv,
    render_command,
)


def plans():
    return build_variant_calling_plans(
        config=load_config(),
        input_bam="data/processed/test.md.bam",
        reference_fasta="data/reference/Egrandis_v2.fa",
        output_gvcf="data/processed/test.g.vcf.gz",
        output_vcf="data/processed/test.vcf.gz",
        filtered_vcf="data/processed/test.filtered.vcf.gz",
        selected_vcf="data/processed/test.pass.snps.vcf.gz",
    )


class VariantCallingCommandTests(TestCase):
    def test_haplotype_caller_reconstructs_selected_parameters(self) -> None:
        argv = build_haplotype_caller_argv(plans().haplotype_caller)

        self.assertEqual(argv[:2], ["gatk", "HaplotypeCaller"])
        self.assertEqual(argv[argv.index("--sample-ploidy") + 1], "4")
        self.assertEqual(argv[argv.index("--emit-ref-confidence") + 1], "GVCF")
        self.assertEqual(
            argv[argv.index("--input") + 1], "data/processed/test.md.bam"
        )

    def test_genotype_gvcfs_omits_unresolved_ploidy(self) -> None:
        plan = plans().genotype_gvcfs
        argv = build_genotype_gvcfs_argv(plan)

        self.assertIsNone(plan.sample_ploidy)
        self.assertNotIn("--sample-ploidy", argv)
        self.assertEqual(
            argv[argv.index("--include-non-variant-sites") + 1], "false"
        )

    def test_genotype_gvcfs_can_render_a_future_explicit_ploidy(self) -> None:
        base = plans().genotype_gvcfs
        explicit = GenotypeGVCFsCommandPlan.model_validate(
            {**base.model_dump(), "sample_ploidy": 4}
        )

        argv = build_genotype_gvcfs_argv(explicit)

        self.assertEqual(argv[argv.index("--sample-ploidy") + 1], "4")

    def test_variant_filtration_matches_all_recorded_filters(self) -> None:
        argv = build_variant_filtration_argv(plans().variant_filtration)
        expressions = [
            argv[index + 1]
            for index, value in enumerate(argv)
            if value == "--filter-expression"
        ]
        names = [
            argv[index + 1]
            for index, value in enumerate(argv)
            if value == "--filter-name"
        ]

        self.assertEqual(
            expressions,
            [
                "QD < 2.0",
                "QUAL < 30.0",
                "SOR > 3.0",
                "FS > 60.0",
                "MQ < 40.0",
                "MQRankSum < -12.5",
                "ReadPosRankSum < -8.0",
            ],
        )
        self.assertEqual(
            names,
            [
                "QD2",
                "QUAL30",
                "SOR3",
                "FS60",
                "MQ40",
                "MQRankSum-12.5",
                "ReadPosRankSum-8",
            ],
        )

    def test_bcftools_selection_matches_recorded_command(self) -> None:
        argv = build_variant_selection_argv(plans().variant_selection)

        self.assertEqual(
            argv,
            [
                "bcftools",
                "view",
                "-f",
                "PASS",
                "-m2",
                "-M2",
                "-v",
                "snps",
                "-Oz",
                "-o",
                "data/processed/test.pass.snps.vcf.gz",
                "data/processed/test.filtered.vcf.gz",
            ],
        )

    def test_plans_reject_input_overwrite(self) -> None:
        base = plans().genotype_gvcfs
        with self.assertRaises(ValidationError):
            GenotypeGVCFsCommandPlan.model_validate(
                {**base.model_dump(), "output_vcf": base.input_gvcf}
            )

    def test_composite_plan_rejects_a_disconnected_chain(self) -> None:
        chain = plans().model_dump()
        chain["genotype_gvcfs"]["input_gvcf"] = Path("other.g.vcf.gz")

        with self.assertRaises(ValidationError):
            VariantCallingPlans.model_validate(chain)

    def test_rendering_quotes_paths_but_never_executes(self) -> None:
        plan = plans().haplotype_caller.model_copy(
            update={"input_bam": Path("data/a sample.bam")}
        )

        rendered = render_command(build_haplotype_caller_argv(plan))

        self.assertIn("'data/a sample.bam'", rendered)

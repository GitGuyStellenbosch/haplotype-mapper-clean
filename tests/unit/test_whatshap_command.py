"""Unit tests for non-executing WhatsHap polyphase command plans."""

from pathlib import Path
from unittest import TestCase

from pydantic import ValidationError

from agent.config import load_config
from pipeline.whatshap import (
    PolyphaseCommandPlan,
    build_polyphase_argv,
    render_polyphase_command,
)


def plan(**updates: object) -> PolyphaseCommandPlan:
    values: dict[str, object] = {
        **load_config().pipeline.whatshap.model_dump(),
        "input_vcf": Path("data/input.filtered.vcf.gz"),
        "phase_inputs": [Path("data/input.bam")],
        "output_vcf": Path("outputs/input.phased.vcf.gz"),
        "reference_fasta": Path("data/reference/Egrandis_v2.fa"),
        "samples": ["sample-1"],
        "chromosomes": ["Chr01"],
    }
    values.update(updates)
    return PolyphaseCommandPlan.model_validate(values)


class WhatsHapCommandTests(TestCase):
    def test_default_plan_contains_explicit_reproducible_parameters(self) -> None:
        argv = build_polyphase_argv(plan())

        self.assertEqual(argv[:2], ["whatshap", "polyphase"])
        self.assertIn("--ploidy", argv)
        self.assertEqual(argv[argv.index("--ploidy") + 1], "4")
        self.assertEqual(
            argv[argv.index("--block-cut-sensitivity") + 1],
            "4",
        )
        self.assertEqual(
            argv[argv.index("--reference") + 1],
            "data/reference/Egrandis_v2.fa",
        )
        self.assertEqual(argv[-2:], ["data/input.filtered.vcf.gz", "data/input.bam"])

    def test_reference_off_omits_reference_argument(self) -> None:
        argv = build_polyphase_argv(
            plan(use_reference=False, reference_fasta=None)
        )

        self.assertNotIn("--reference", argv)

    def test_block_cut_sensitivity_is_limited_to_documented_range(self) -> None:
        with self.assertRaises(ValidationError):
            plan(block_cut_sensitivity=6)

    def test_reference_on_requires_reference_path(self) -> None:
        with self.assertRaises(ValidationError):
            plan(reference_fasta=None)

    def test_input_cannot_be_overwritten_by_output(self) -> None:
        with self.assertRaises(ValidationError):
            plan(output_vcf=Path("data/input.filtered.vcf.gz"))

    def test_human_rendering_quotes_paths_without_changing_argv(self) -> None:
        spaced = plan(
            input_vcf=Path("data/a sample.vcf.gz"),
            phase_inputs=[Path("data/a sample.bam")],
        )

        rendered = render_polyphase_command(spaced)

        self.assertIn("'data/a sample.vcf.gz'", rendered)
        self.assertIn("'data/a sample.bam'", rendered)

    def test_documented_optional_flags_are_rendered_explicitly(self) -> None:
        argv = build_polyphase_argv(
            plan(
                block_cut_sensitivity=2,
                include_haploid_sets=True,
                use_supplementary=True,
                supplementary_distance=50000,
            )
        )

        self.assertIn("--include-haploid-sets", argv)
        self.assertIn("--use-supplementary", argv)
        self.assertEqual(
            argv[argv.index("--supplementary-distance") + 1],
            "50000",
        )

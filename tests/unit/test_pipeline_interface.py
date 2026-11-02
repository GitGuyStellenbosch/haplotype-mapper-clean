"""Unit tests for the deterministic phasing boundary."""

import gzip
import struct
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from agent.pipeline_interface import (
    InputValidationError,
    MockPhasingBackend,
    PhasingStrategy,
    PipelineInterface,
    StrategyError,
    inspect_real_inputs,
    validate_real_inputs,
)
from agent.state import (
    InputFiles,
    PhasingAttempt,
    PhasingEvidence,
    PhasingOutcome,
    RegionIdentity,
    RegionState,
)


def example_state(
    *,
    inputs: InputFiles | None = None,
    attempts: list[PhasingAttempt] | None = None,
) -> RegionState:
    return RegionState(
        sample_id="synthetic-sample",
        region=RegionIdentity(
            gene="Eucgr.A00001",
            target_region="Gene start region",
            chromosome="Chr01",
            start_position=100,
            end_position=120,
        ),
        inputs=inputs or InputFiles(),
        attempt_history=attempts or [],
    )


class PipelineInterfaceTests(TestCase):
    def _write_real_input_fixture(
        self,
        root: Path,
        *,
        bam_sample: str = "SAMPLE1",
        vcf_sample: str = "SAMPLE1",
        genotype: str = "0/0/1/1",
        bam_contig_length: int = 1000,
        include_read_group: bool = True,
    ) -> InputFiles:
        bam = root / "sample.bam"
        vcf = root / "sample.filtered.vcf.gz"
        fasta = root / "reference.fa"
        header = f"@HD\tVN:1.6\n@SQ\tSN:Chr01\tLN:{bam_contig_length}\n"
        if include_read_group:
            header += f"@RG\tID:rg1\tSM:{bam_sample}\n"
        with gzip.open(bam, "wb") as handle:
            encoded = header.encode("utf-8")
            handle.write(b"BAM\x01" + struct.pack("<i", len(encoded)) + encoded)
        Path(f"{bam}.bai").write_bytes(b"synthetic-index")

        with gzip.open(vcf, "wt", encoding="utf-8") as handle:
            handle.write(
                "##fileformat=VCFv4.2\n"
                "##contig=<ID=Chr01,length=1000>\n"
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=Genotype>\n"
                f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{vcf_sample}\n"
                f"Chr01\t10\t.\tA\tC\t60\tPASS\t.\tGT\t{genotype}\n"
            )
        Path(f"{vcf}.tbi").write_bytes(b"synthetic-index")

        fasta.write_text(">Chr01\n" + "A" * 1000 + "\n", encoding="utf-8")
        Path(f"{fasta}.fai").write_text(
            "Chr01\t1000\t7\t1000\t1001\n", encoding="utf-8"
        )
        fasta.with_suffix(".dict").write_text(
            "@HD\tVN:1.6\n@SQ\tSN:Chr01\tLN:1000\n", encoding="utf-8"
        )
        return InputFiles(bam=bam, vcf=vcf, reference_fasta=fasta)

    def test_mock_backend_creates_first_attempt(self) -> None:
        backend = MockPhasingBackend(
            {
                "whatshap-default": PhasingEvidence(
                    outcome=PhasingOutcome.PHASED,
                    haplotype_count=2,
                )
            }
        )
        pipeline = PipelineInterface(backend, validate_inputs=False)

        attempt = pipeline.run(
            example_state(),
            PhasingStrategy(
                strategy_id="whatshap-default",
                parameters={"block_cut_sensitivity": 4},
            ),
        )

        self.assertEqual(attempt.attempt_number, 1)
        self.assertEqual(attempt.evidence.outcome, PhasingOutcome.PHASED)
        self.assertEqual(len(backend.requests), 1)

    def test_existing_attempt_increments_attempt_number(self) -> None:
        first_attempt = PhasingAttempt(
            attempt_number=1,
            strategy_id="whatshap-default",
            evidence=PhasingEvidence(outcome=PhasingOutcome.FAILED),
        )
        backend = MockPhasingBackend(
            {
                "whatshap-retry-1": PhasingEvidence(
                    outcome=PhasingOutcome.PARTIALLY_PHASED
                )
            }
        )
        pipeline = PipelineInterface(backend, validate_inputs=False)

        attempt = pipeline.run(
            example_state(attempts=[first_attempt]),
            PhasingStrategy(strategy_id="whatshap-retry-1"),
        )

        self.assertEqual(attempt.attempt_number, 2)

    def test_repeated_strategy_is_rejected_before_backend_call(self) -> None:
        first_attempt = PhasingAttempt(
            attempt_number=1,
            strategy_id="whatshap-default",
            evidence=PhasingEvidence(outcome=PhasingOutcome.FAILED),
        )
        backend = MockPhasingBackend(
            {"whatshap-default": PhasingEvidence(outcome=PhasingOutcome.PHASED)}
        )
        pipeline = PipelineInterface(backend, validate_inputs=False)

        with self.assertRaises(StrategyError):
            pipeline.run(
                example_state(attempts=[first_attempt]),
                PhasingStrategy(strategy_id="whatshap-default"),
            )

        self.assertEqual(backend.requests, [])

    def test_unknown_synthetic_strategy_has_clear_error(self) -> None:
        pipeline = PipelineInterface(MockPhasingBackend({}), validate_inputs=False)

        with self.assertRaisesRegex(StrategyError, "no synthetic result"):
            pipeline.run(
                example_state(),
                PhasingStrategy(strategy_id="not-configured"),
            )

    def test_missing_real_inputs_are_reported_together(self) -> None:
        report = inspect_real_inputs(InputFiles())

        self.assertFalse(report.is_valid)
        self.assertEqual(len(report.errors), 3)
        with self.assertRaises(InputValidationError):
            validate_real_inputs(InputFiles())

    def test_compatible_files_pass_content_aware_preflight(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report = validate_real_inputs(
                self._write_real_input_fixture(root),
                expected_sample="SAMPLE1",
                expected_ploidy=4,
            )

            self.assertTrue(report.is_valid)
            self.assertEqual(
                set(report.discovered_indexes),
                {"bam", "vcf", "reference_fasta", "reference_dictionary"},
            )
            self.assertEqual(report.vcf_samples, ["SAMPLE1"])
            self.assertEqual(report.bam_read_group_samples, ["SAMPLE1"])
            self.assertEqual(report.observed_genotype_ploidies, [4])
            self.assertEqual(report.contig_counts["bam"], 1)

    def test_sample_ploidy_and_contig_mismatches_are_reported_together(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            inputs = self._write_real_input_fixture(
                Path(temporary_directory),
                bam_sample="OTHER",
                genotype="0/1",
                bam_contig_length=999,
            )
            report = inspect_real_inputs(
                inputs,
                expected_sample="SAMPLE1",
                expected_ploidy=4,
            )

        self.assertFalse(report.is_valid)
        joined = "; ".join(report.errors)
        self.assertIn("BAM read-group samples", joined)
        self.assertIn("genotype ploidy 2", joined)
        self.assertIn("BAM/reference length mismatches", joined)

    def test_missing_bam_read_groups_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            inputs = self._write_real_input_fixture(
                Path(temporary_directory), include_read_group=False
            )
            report = inspect_real_inputs(inputs, expected_sample="SAMPLE1")

        self.assertIn("BAM header has no read groups", report.errors)

    def test_expected_sample_must_exist_in_vcf(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            inputs = self._write_real_input_fixture(
                Path(temporary_directory), vcf_sample="OTHER"
            )
            report = inspect_real_inputs(inputs, expected_sample="SAMPLE1")

        self.assertTrue(
            any("VCF samples do not contain expected sample" in error for error in report.errors)
        )

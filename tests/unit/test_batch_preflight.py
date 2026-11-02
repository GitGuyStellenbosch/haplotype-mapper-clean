"""Tests for the small multi-sample preflight adapter."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from pipeline.batch_preflight import (
    BatchManifestError,
    load_batch_manifest,
    validate_batch_manifest,
)
from pipeline.preflight import InputValidationReport


HEADER = (
    "batch_id,sample_id,bam,bam_index,phased_vcf,vcf_index,reference_fasta,"
    "reference_fai,reference_dictionary\n"
)


class BatchPreflightTests(TestCase):
    def test_duplicate_sample_ids_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "manifest.csv"
            row = "batch,S1,a.bam,a.bai,a.vcf.gz,a.tbi,ref.fa,ref.fai,ref.dict\n"
            manifest.write_text(HEADER + row + row, encoding="utf-8")

            with self.assertRaisesRegex(BatchManifestError, "unique"):
                load_batch_manifest(manifest)

    def test_valid_manifest_reuses_existing_preflight(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            names = [
                "sample.bam",
                "sample.bam.bai",
                "sample.vcf.gz",
                "sample.vcf.gz.tbi",
                "reference.fa",
                "reference.fa.fai",
                "reference.dict",
            ]
            for name in names:
                (root / name).write_bytes(b"fixture")
            manifest = root / "manifest.csv"
            manifest.write_text(
                HEADER
                + "batch,S1,sample.bam,sample.bam.bai,sample.vcf.gz,"
                "sample.vcf.gz.tbi,reference.fa,reference.fa.fai,reference.dict\n",
                encoding="utf-8",
            )
            discovered = {
                "bam": root / "sample.bam.bai",
                "vcf": root / "sample.vcf.gz.tbi",
                "reference_fasta": root / "reference.fa.fai",
                "reference_dictionary": root / "reference.dict",
            }
            preflight = InputValidationReport(
                checks_completed=["fixture"], discovered_indexes=discovered
            )

            with patch(
                "pipeline.batch_preflight.inspect_real_inputs",
                return_value=preflight,
            ) as inspect:
                report = validate_batch_manifest(manifest, expected_ploidy=4)

            self.assertEqual(report["summary"], {"samples": 1, "passed": 1, "failed": 0})
            inspect.assert_called_once()
            self.assertEqual(inspect.call_args.kwargs["expected_sample"], "S1")
            self.assertEqual(inspect.call_args.kwargs["expected_ploidy"], 4)

"""Tests for extracting typed evidence from a phased one-sample VCF."""

from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

from agent.state import PhasingOutcome
from evaluation.real_evidence_census import summarize_states
from pipeline.real_evidence import (
    REAL_EVIDENCE_DATA_STATUS,
    RealEvidenceError,
    build_real_region_states,
    load_target_catalogue,
)


class RealEvidenceExtractionTests(unittest.TestCase):
    def _write_catalogue(self, root: Path, *, mismatch: bool = False) -> tuple[Path, Path]:
        bed = root / "targets.bed"
        bed.write_text(
            "Chr01\t9\t20\nChr01\t29\t40\nChr01\t49\t60\n",
            encoding="utf-8",
        )
        csv_path = root / "regions.csv"
        second_start = 31 if mismatch else 30
        csv_path.write_text(
            "Gene,Target Region,Chromosome,5' Position,3'Position,Category,Target size\n"
            "GeneA,Gene start region,Chr01,10,20,1,11\n"
            f"GeneB,Gene end region,Chr01,{second_start},40,2,11\n"
            "GeneC,Upstream region,Chr01,50,60,3,11\n",
            encoding="utf-8",
        )
        return bed, csv_path

    def _write_vcf(self, root: Path) -> Path:
        path = root / "phased.vcf.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(
                "##fileformat=VCFv4.2\n"
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=Genotype>\n"
                "##FORMAT=<ID=PS,Number=1,Type=Integer,Description=Phase set>\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\n"
                "Chr01\t10\t.\tA\tC\t60\tPASS\t.\tGT:PS\t0|0|1|1:10\n"
                "Chr01\t15\t.\tG\tT\t60\tPASS\t.\tGT:PS\t1/1/1/1:.\n"
                "Chr01\t35\t.\tC\tT\t60\tPASS\t.\tGT:PS\t0/1/1/1:.\n"
            )
        return path

    def test_covered_regions_become_real_unlabeled_states(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            bed, categories = self._write_catalogue(root)
            vcf = self._write_vcf(root)
            states = build_real_region_states(
                vcf_path=vcf,
                bam_path=root / "sample.bam",
                bed_path=bed,
                region_categories_path=categories,
                reference_fasta=root / "reference.fa",
                expected_sample="SAMPLE1",
            )

        self.assertEqual(len(states), 2)
        first = states[0]
        self.assertEqual(
            first.state.region.region_key,
            "GeneA|Gene start region|Chr01|10|20",
        )
        self.assertEqual(first.state.evidence.heterozygous_snp_count, 1)
        self.assertEqual(first.state.evidence.phased_snp_count, 1)
        self.assertEqual(first.state.evidence.unphased_snp_count, 0)
        self.assertIsNone(first.state.evidence.block_count)
        self.assertIsNone(first.state.evidence.block_n50)
        self.assertEqual(first.state.evidence.outcome, PhasingOutcome.PHASED)
        self.assertEqual(first.state.evidence.additional_metrics["total_snp_count"], 2)
        self.assertFalse(first.state.evidence.family_data_available)
        self.assertEqual(first.state.metadata["data_status"], REAL_EVIDENCE_DATA_STATUS)
        self.assertNotIn("catalogue_category", first.state.metadata)

        second = states[1].state
        self.assertEqual(second.evidence.outcome, PhasingOutcome.UNPHASED)
        self.assertEqual(second.evidence.heterozygous_snp_count, 1)
        self.assertEqual(second.evidence.phased_snp_count, 0)
        self.assertEqual(second.evidence.unphased_snp_count, 1)

    def test_bed_and_catalogue_coordinates_must_match_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            bed, categories = self._write_catalogue(
                Path(temp_directory), mismatch=True
            )
            with self.assertRaisesRegex(RealEvidenceError, "coordinate mismatch"):
                load_target_catalogue(bed, categories)

    def test_census_summarizes_existing_states_without_category_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            bed, categories = self._write_catalogue(root)
            states = build_real_region_states(
                vcf_path=self._write_vcf(root),
                bam_path=root / "sample.bam",
                bed_path=bed,
                region_categories_path=categories,
                reference_fasta=root / "reference.fa",
                expected_sample="SAMPLE1",
            )

        summary = summarize_states(states)

        self.assertEqual(summary["covered_regions"], 2)
        self.assertEqual(summary["outcomes"], {"phased": 1, "unphased": 1})
        self.assertEqual(summary["snp_totals"]["heterozygous_snps"], 2)
        self.assertEqual(summary["catalogue_categories_external_only"], {"1": 1, "2": 1})
        self.assertEqual(summary["assessor_category_leaks"], 0)


if __name__ == "__main__":
    unittest.main()

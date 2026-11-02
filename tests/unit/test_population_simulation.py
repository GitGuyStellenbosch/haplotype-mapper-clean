"""Tests for frequency-grounded genotype generation and its non-scoring pilot."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

from openpyxl import Workbook
from pydantic import ValidationError

from agent.config import load_config
from agent.nodes import AssessmentResult
from agent.state import DecisionAction, DecisionSource
from evaluation.population_pilot import (
    PopulationPilotAudit,
    render_pilot_summary,
    run_population_pilot,
)
from evaluation.population_simulation import (
    POPULATION_SIMULATION_DATA_STATUS,
    CatalogueHaplotype,
    CatalogueRegion,
    PopulationSimulationCase,
    build_population_case,
    generate_population_dataset,
    iter_population_dataset,
    load_population_cases,
)
from agent.state import RegionIdentity


class AcceptingAssessor:
    def assess(self, state):
        return AssessmentResult(
            action=DecisionAction.ACCEPT,
            confidence=0.8,
            rationale="Test fixture accepted the supplied perfectly observed evidence.",
            source=DecisionSource.LLM,
        )


def _region(category: int) -> CatalogueRegion:
    return CatalogueRegion(
        identity=RegionIdentity(
            gene=f"Gene{category}",
            target_region="start",
            chromosome="Chr01",
            start_position=category * 100,
            end_position=category * 100 + 3,
        ),
        category=category,
        snp_count=4,
        haplotypes=(
            CatalogueHaplotype("h1", "AAAA", 0.5),
            CatalogueHaplotype("h2", "AATT", 0.5),
        ),
    )


def _case_with_zygosity(category: int, zygosity: str) -> PopulationSimulationCase:
    for number in range(1, 1000):
        case = build_population_case(
            _region(category),
            individual_id=f"DEV_{category}_{zygosity}_{number}",
            partition="development",
            seed=20260907,
        )
        if case.truth.zygosity == zygosity:
            return case
    raise AssertionError(f"could not construct {zygosity} test case")


class PopulationSimulationTests(TestCase):
    def _write_workbook(self, path: Path) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Notes"
        data = workbook.create_sheet("Supplementary File 1")
        data.append(["Catalogue"])
        data.append(["Haplotype frequency"])
        columns = [
            "Gene",
            "Target Region",
            "Category",
            "Chromosome",
            "5' Position",
            "3'Position",
            "Haplotype ID",
            "SNP Haplotype",
            "Number of SNPs",
            "E. grandis",
        ]
        data.append(columns)
        data.append(["Gene1", "start", 1, "Chr01", 10, 11, "h1", "AA", 2, 0.75])
        data.append(["Gene1", "start", 1, "Chr01", 10, 11, "h2", "AT", 2, 0.25])
        data.append(["Gene2", "end", 2, "Chr01", 20, 21, "h1", "CC", 2, 1.0])
        data.append(["Gene3", "downstream", 3, "Chr01", 30, 31, "h1", "GG", 2, 0.0])
        workbook.save(path)

    def test_generation_is_reproducible_tagged_and_excludes_zero_sum_region(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workbook = root / "catalogue.xlsx"
            self._write_workbook(workbook)
            first = generate_population_dataset(
                workbook_path=workbook,
                output_directory=root / "first",
                seed=42,
                individual_count=3,
                pilot_per_stratum=0,
            )
            second = generate_population_dataset(
                workbook_path=workbook,
                output_directory=root / "second",
                seed=42,
                individual_count=3,
                pilot_per_stratum=0,
            )
            cases = list(iter_population_dataset(first.cases_path))
            pilot_cases = load_population_cases(first.pilot_cases_path)

        self.assertEqual(first.generated_case_count, 6)
        self.assertEqual(pilot_cases, [])
        self.assertEqual(first.cases_sha256, second.cases_sha256)
        self.assertEqual(len(first.excluded_regions), 1)
        self.assertEqual(
            first.excluded_regions[0].reason,
            "zero_population_frequency",
        )
        self.assertTrue(
            all(case.data_status == POPULATION_SIMULATION_DATA_STATUS for case in cases)
        )
        self.assertTrue(
            all(
                case.state.metadata["data_status"]
                == POPULATION_SIMULATION_DATA_STATUS
                for case in cases
            )
        )

    def test_haplotype_differences_drive_existing_evidence_shape(self) -> None:
        heterozygous = _case_with_zygosity(1, "heterozygous")
        homozygous = _case_with_zygosity(1, "homozygous")

        self.assertEqual(heterozygous.truth.heterozygous_snp_count, 2)
        self.assertEqual(heterozygous.state.evidence.phased_snp_count, 2)
        self.assertEqual(heterozygous.state.evidence.unphased_snp_count, 0)
        self.assertEqual(heterozygous.state.evidence.block_count, 1)
        self.assertIsNone(heterozygous.state.evidence.block_n50)
        self.assertEqual(homozygous.truth.heterozygous_snp_count, 0)
        self.assertEqual(homozygous.state.evidence.block_count, 0)
        self.assertFalse(homozygous.truth.action_ground_truth_available)

    def test_simulation_status_cannot_be_changed_to_synthetic(self) -> None:
        case = _case_with_zygosity(1, "heterozygous")
        payload = case.model_dump()
        payload["data_status"] = "synthetic"
        with self.assertRaises(ValidationError):
            PopulationSimulationCase.model_validate(payload)

    def test_pilot_runs_all_six_strata_without_scoring(self) -> None:
        config = load_config()
        # This historical population pilot deliberately has no executable retries.
        config = config.model_copy(update={"pipeline": config.pipeline.model_copy(
            update={"retry_strategies": []}
        )})
        cases = [
            _case_with_zygosity(category, zygosity)
            for category in (1, 2, 3)
            for zygosity in ("homozygous", "heterozygous")
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = run_population_pilot(
                config=config,
                cases=cases,
                output_directory=temporary_directory,
                assessor=AcceptingAssessor(),
            )
            summary = render_pilot_summary(report)
            audit_path = next((Path(temporary_directory) / "pilot_audits").iterdir())
            audit = PopulationPilotAudit.model_validate_json(audit_path.read_text())

        self.assertEqual(report.processed_cases, 6)
        self.assertEqual(len(report.decisions_by_stratum), 6)
        self.assertEqual(report.action_counts["accept"], 6)
        self.assertFalse(report.scoring_performed)
        self.assertIsNone(report.accuracy)
        self.assertFalse(audit.action_ground_truth_available)
        self.assertIn("Accuracy: not calculated", summary)

        audit_payload = audit.model_dump()
        audit_payload["scoring_performed"] = True
        with self.assertRaises(ValidationError):
            PopulationPilotAudit.model_validate(audit_payload)


if __name__ == "__main__":
    import unittest

    unittest.main()

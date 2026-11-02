"""Tests for deterministic real-panel selection and storage."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from agent.state import (
    InputFiles,
    PhasingAttempt,
    PhasingEvidence,
    PhasingOutcome,
    RegionIdentity,
    RegionState,
)
from evaluation.real_panel import load_panel, select_sample_cases, write_panel
from evaluation.real_panel_run import choose_rationale_review_ids
from pipeline.real_evidence import REAL_EVIDENCE_DATA_STATUS, ExtractedRegionState


def extracted(outcome: PhasingOutcome, position: int, call_rate: float) -> ExtractedRegionState:
    evidence = PhasingEvidence(
        outcome=outcome,
        heterozygous_snp_count=0 if outcome == PhasingOutcome.NO_PHASE_INFORMATIVE_VARIANTS else 1,
        phased_snp_count=1 if outcome == PhasingOutcome.PHASED else 0,
        unphased_snp_count=1 if outcome == PhasingOutcome.UNPHASED else 0,
        call_rate=call_rate,
    )
    state = RegionState(
        sample_id="S1",
        region=RegionIdentity(
            gene=f"Gene{position}",
            target_region="start",
            chromosome="Chr01",
            start_position=position,
            end_position=position,
        ),
        inputs=InputFiles(),
        evidence=evidence,
        attempt_history=[
            PhasingAttempt(attempt_number=1, strategy_id="provided", evidence=evidence)
        ],
        metadata={"data_status": REAL_EVIDENCE_DATA_STATUS},
    )
    return ExtractedRegionState(state=state, catalogue_category="1", catalogue_row=position)


class RealPanelTests(TestCase):
    def test_selection_is_deterministic_and_covers_three_outcomes(self) -> None:
        states = [
            extracted(PhasingOutcome.NO_PHASE_INFORMATIVE_VARIANTS, 10, 0),
            extracted(PhasingOutcome.NO_PHASE_INFORMATIVE_VARIANTS, 20, 1),
            extracted(PhasingOutcome.PHASED, 30, 1),
            extracted(PhasingOutcome.UNPHASED, 40, 1),
        ]

        first = select_sample_cases(states, sample_index=1, seed=20260915)
        second = select_sample_cases(states, sample_index=1, seed=20260915)

        self.assertEqual(first, second)
        self.assertEqual(
            [item.state.evidence.outcome for _, item in first],
            [
                PhasingOutcome.NO_PHASE_INFORMATIVE_VARIANTS,
                PhasingOutcome.PHASED,
                PhasingOutcome.UNPHASED,
            ],
        )

    def test_written_panel_round_trips_through_region_state(self) -> None:
        item = extracted(PhasingOutcome.PHASED, 30, 1)
        record = {
            "case_id": "S1::case",
            "state": item.state.model_dump(mode="json", exclude_computed_fields=True),
        }
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            write_panel([record], {"case_count": 1}, output)
            loaded = load_panel(output / "panel.jsonl")

        self.assertEqual(loaded[0]["state"], item.state)

    def test_rationale_review_is_frozen_across_outcomes_and_call_rates(self) -> None:
        panel = [
            {
                "case_id": f"case-{index}",
                "evidence_outcome": outcome,
                "call_rate_bin": call_rate,
            }
            for index, (outcome, call_rate) in enumerate(
                (
                    ("no_phase_informative_variants", "complete"),
                    ("phased", "complete"),
                    ("partially_phased", "complete"),
                    ("unphased", "complete"),
                    ("no_phase_informative_variants", "zero"),
                    ("no_phase_informative_variants", "partial"),
                )
            )
        ]

        selected = choose_rationale_review_ids(panel)

        self.assertEqual(len(selected), 6)
        self.assertEqual(len(set(selected)), 6)

"""Tests for the frozen real-panel repeatability runner."""

from types import SimpleNamespace
from unittest import TestCase

from agent.state import (
    DecisionAction,
    DecisionRecord,
    DecisionSource,
    PhasingEvidence,
    PhasingOutcome,
    RegionIdentity,
    RegionState,
)
from evaluation.real_repeatability import build_report, build_trials


def panel_item(case_id: str, position: int) -> dict[str, object]:
    sample_id, _ = case_id.split("::", 1)
    return {
        "case_id": case_id,
        "evidence_outcome": "phased",
        "catalogue_category_external": "1",
        "catalogue_row": position,
        "state": RegionState(
            sample_id=sample_id,
            region=RegionIdentity(
                gene=f"gene-{position}",
                target_region="start",
                chromosome="Chr01",
                start_position=position,
                end_position=position,
            ),
            evidence=PhasingEvidence(outcome=PhasingOutcome.PHASED),
            metadata={"data_status": "real_evidence_no_ground_truth"},
        ),
    }


def audit(action: str | None, confidence: float = 1.0) -> SimpleNamespace:
    if action is None:
        return SimpleNamespace(
            run_id="error",
            duration_seconds=2.0,
            execution_status="error",
            final_state=None,
            error_type="ModelAssessmentError",
            error_message="invalid structured output",
        )
    decision = DecisionRecord(
        action=DecisionAction(action),
        confidence=confidence,
        rationale="Phased heterozygous evidence supports manual review.",
        source=DecisionSource.LLM,
    )
    return SimpleNamespace(
        run_id=f"run-{action}-{confidence}",
        duration_seconds=1.0,
        execution_status="success",
        final_state=SimpleNamespace(decision_history=[decision]),
        error_type=None,
        error_message=None,
    )


class RealRepeatabilityTests(TestCase):
    def test_trials_are_round_major_and_preserve_state_fingerprints(self) -> None:
        case_ids = ("S1::case-1", "S2::case-2")
        panel = [panel_item(case_id, index) for index, case_id in enumerate(case_ids, 1)]

        extracted, trials = build_trials(panel, case_ids)

        self.assertEqual(len(extracted), 6)
        self.assertEqual(
            [(item["repetition"], item["case_id"]) for item in trials],
            [(repetition, case_id) for repetition in (1, 2, 3) for case_id in case_ids],
        )
        for case_id in case_ids:
            fingerprints = {
                item["state_sha256"] for item in trials if item["case_id"] == case_id
            }
            self.assertEqual(len(fingerprints), 1)

    def test_report_separates_agreement_confidence_and_failures(self) -> None:
        trials = [
            {
                "trial_index": index + 1,
                "repetition": index // 2 + 1,
                "case_id": ("stable" if index % 2 == 0 else "variable"),
                "evidence_outcome": "phased",
                "state_sha256": "same",
            }
            for index in range(6)
        ]
        records = [
            audit("accept", 0.9),
            audit("accept", 0.4),
            audit("accept", 0.9),
            audit("escalate", 0.8),
            audit("accept", 0.9),
            audit(None),
        ]

        report = build_report(records, trials)

        self.assertEqual(report["strict_three_run_action_agreement"]["cases"], 1)
        self.assertEqual(
            report["pairwise_action_agreement_among_valid_calls"]["proportion"],
            0.75,
        )
        self.assertEqual(report["failed_calls"], 1)
        self.assertEqual(report["confidence_variation"]["cases_with_nonzero_range"], 1)
        self.assertFalse(report["phase1_always_accept_pattern_persisted"])


if __name__ == "__main__":
    raise SystemExit(TestCase)

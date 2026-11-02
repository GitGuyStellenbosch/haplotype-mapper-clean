"""Tests for non-evaluation real-evidence audit records and summaries."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from agent.real_smoke import (
    RealEvidenceAuditRecord,
    render_smoke_summary,
    write_real_audit,
)
from agent.state import (
    DecisionAction,
    DecisionRecord,
    DecisionSource,
    PhasingAttempt,
    PhasingEvidence,
    PhasingOutcome,
    RegionIdentity,
    RegionState,
)
from pipeline.real_evidence import REAL_EVIDENCE_DATA_STATUS


def _state(*, decided: bool) -> RegionState:
    evidence = PhasingEvidence(
        outcome=PhasingOutcome.PHASED,
        heterozygous_snp_count=1,
        phased_snp_count=1,
        unphased_snp_count=0,
        family_data_available=False,
    )
    attempt = PhasingAttempt(
        attempt_number=1,
        strategy_id="provided-whatshap-polyphase-output",
        evidence=evidence,
    )
    decision = (
        DecisionRecord(
            action=DecisionAction.ESCALATE,
            confidence=1.0,
            rationale="Unlabeled real evidence requires expert review.",
            source=DecisionSource.LLM,
        )
        if decided
        else None
    )
    return RegionState(
        sample_id="SAMPLE1",
        region=RegionIdentity(
            gene="GeneA",
            target_region="start",
            chromosome="Chr01",
            start_position=10,
            end_position=20,
        ),
        evidence=evidence,
        attempt_history=[attempt],
        decision=decision,
        decision_history=[decision] if decision is not None else [],
        metadata={"data_status": REAL_EVIDENCE_DATA_STATUS},
    )


def _record() -> RealEvidenceAuditRecord:
    initial = _state(decided=False)
    final = _state(decided=True)
    return RealEvidenceAuditRecord(
        run_id="run-1",
        batch_id="batch-1",
        recorded_at=datetime.now(timezone.utc),
        data_status=REAL_EVIDENCE_DATA_STATUS,
        execution_status="success",
        sample_id="SAMPLE1",
        region_key=initial.region.region_key,
        model_provider="ollama",
        model_name="test-model",
        duration_seconds=1.0,
        configuration={"pipeline": {"mode": "real"}},
        initial_state=initial,
        final_state=final,
    )


class RealEvidenceAuditTests(unittest.TestCase):
    def test_audit_cannot_be_tagged_as_synthetic_or_scored(self) -> None:
        data = _record().model_dump()
        data["data_status"] = "synthetic"
        with self.assertRaises(ValidationError):
            RealEvidenceAuditRecord.model_validate(data)

        data = _record().model_dump()
        data["scoring_performed"] = True
        with self.assertRaises(ValidationError):
            RealEvidenceAuditRecord.model_validate(data)

    def test_written_audit_round_trips_and_summary_is_non_scoring(self) -> None:
        record = _record()
        with tempfile.TemporaryDirectory() as temp_directory:
            output = Path(temp_directory) / "region.json"
            write_real_audit(record, output)
            loaded = RealEvidenceAuditRecord.model_validate_json(
                output.read_text(encoding="utf-8")
            )
            summary = render_smoke_summary(
                batch_id="batch-1",
                sample_id="SAMPLE1",
                output_directory=Path(temp_directory),
                total_covered_regions=1,
                records=[loaded],
            )

        self.assertEqual(loaded.data_status, REAL_EVIDENCE_DATA_STATUS)
        self.assertIn("Scoring performed: no", summary)
        self.assertIn("escalate: 1", summary)
        self.assertIn("Initial decisions (all sources):", summary)
        self.assertIn("deterministic: 0", summary)
        self.assertIn("model: 1", summary)
        self.assertIn("Unlabeled real evidence", summary)


if __name__ == "__main__":
    unittest.main()

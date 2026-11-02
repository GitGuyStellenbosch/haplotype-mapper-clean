"""Unit tests for the shared agent state contract."""

from pathlib import Path
from unittest import TestCase

from pydantic import ValidationError

from agent.state import (
    DecisionAction,
    DecisionRecord,
    DecisionSource,
    InputFiles,
    PhasingAttempt,
    PhasingEvidence,
    PhasingOutcome,
    RegionIdentity,
    RegionState,
)


def example_region() -> RegionIdentity:
    return RegionIdentity(
        gene="Eucgr.A00001",
        target_region="Gene start region",
        chromosome="Chr01",
        start_position=100,
        end_position=100,
    )


class RegionStateTests(TestCase):
    def test_minimal_preparation_state_is_valid(self) -> None:
        state = RegionState(sample_id="synthetic-sample", region=example_region())

        self.assertEqual(state.region.region_key, "Eucgr.A00001|Gene start region|Chr01|100|100")
        self.assertEqual(state.evidence.outcome, PhasingOutcome.NOT_RUN)
        self.assertEqual(state.retries_used, 0)

    def test_phase_block_statistics_are_typed_evidence(self) -> None:
        evidence = PhasingEvidence(block_count=3, block_n50=1250)

        self.assertEqual(evidence.block_count, 3)
        self.assertEqual(evidence.block_n50, 1250)
        with self.assertRaises(ValueError):
            PhasingEvidence(block_count=-1)

    def test_paths_need_not_exist_during_preparation(self) -> None:
        inputs = InputFiles(
            bam=Path("data/synthetic/example.bam"),
            vcf=Path("data/synthetic/example.vcf.gz"),
        )

        state = RegionState(
            sample_id="synthetic-sample",
            region=example_region(),
            inputs=inputs,
        )

        self.assertEqual(state.inputs.bam, Path("data/synthetic/example.bam"))

    def test_call_rate_must_be_normalized(self) -> None:
        with self.assertRaises(ValidationError):
            PhasingEvidence(call_rate=80.0)

    def test_mendelian_errors_require_family_context(self) -> None:
        with self.assertRaises(ValidationError):
            PhasingEvidence(mendelian_errors=1)

    def test_single_snp_region_is_valid(self) -> None:
        region = example_region()

        self.assertEqual(region.start_position, region.end_position)

    def test_reversed_region_is_invalid(self) -> None:
        with self.assertRaises(ValidationError):
            RegionIdentity(
                gene="Eucgr.A00001",
                target_region="Gene start region",
                chromosome="Chr01",
                start_position=101,
                end_position=100,
            )

    def test_repeated_strategy_is_rejected(self) -> None:
        attempt = PhasingAttempt(
            attempt_number=1,
            strategy_id="whatshap-default",
            evidence=PhasingEvidence(outcome=PhasingOutcome.FAILED),
        )

        with self.assertRaises(ValidationError):
            RegionState(
                sample_id="synthetic-sample",
                region=example_region(),
                attempt_history=[
                    attempt,
                    attempt.model_copy(update={"attempt_number": 2}),
                ],
            )

    def test_only_closed_decision_actions_are_accepted(self) -> None:
        decision = DecisionRecord(
            action=DecisionAction.ESCALATE,
            confidence=0.9,
            rationale="Synthetic example with insufficient evidence.",
            source=DecisionSource.DETERMINISTIC,
        )

        state = RegionState(
            sample_id="synthetic-sample",
            region=example_region(),
            decision=decision,
        )

        self.assertEqual(state.decision.action, DecisionAction.ESCALATE)

        with self.assertRaises(ValidationError):
            DecisionRecord(
                action="discard",
                confidence=0.9,
                rationale="This action is outside the project decision space.",
                source=DecisionSource.DETERMINISTIC,
            )

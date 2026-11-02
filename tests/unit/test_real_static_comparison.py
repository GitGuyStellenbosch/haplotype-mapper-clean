"""Boundary checks for the frozen Phase 14 research comparator."""

from unittest import TestCase

from agent.state import DecisionAction, PhasingEvidence, PhasingOutcome
from evaluation.real_static_comparison import build_static_assessor
from tests.unit.test_nodes import example_state


class RealStaticComparisonTests(TestCase):
    def test_accept_requires_all_declared_evidence(self) -> None:
        state = example_state()
        state.evidence = PhasingEvidence(
            outcome=PhasingOutcome.PHASED,
            heterozygous_snp_count=2,
            phased_snp_count=2,
            unphased_snp_count=0,
            call_rate=1.0,
            additional_metrics={"phase_set_count": 1},
        )
        assessor = build_static_assessor()
        self.assertEqual(assessor.assess(state).action, DecisionAction.ACCEPT)

        for change in (
            {"heterozygous_snp_count": 1, "phased_snp_count": 1},
            {"unphased_snp_count": 1},
            {"call_rate": None},
            {"call_rate": 0.9},
            {"additional_metrics": {"phase_set_count": 2}},
            {"additional_metrics": {}},
            {"outcome": PhasingOutcome.PARTIALLY_PHASED},
        ):
            altered = state.model_copy(deep=True)
            altered.evidence = state.evidence.model_copy(update=change)
            self.assertEqual(assessor.assess(altered).action, DecisionAction.ESCALATE)

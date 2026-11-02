"""Unit tests for the versioned fixed-rule baseline interface."""

from unittest import TestCase

from agent.config import StaticBaselineConfig
from agent.state import (
    DecisionAction,
    DecisionSource,
    PhasingEvidence,
    PhasingOutcome,
    PhasingStrategy,
)
from agent.static_baseline import (
    PredicateRule,
    RuleProvenance,
    StaticBaselineAssessor,
    StaticBaselineError,
    StaticRuleDecision,
)
from tests.unit.test_nodes import example_state


def definition() -> StaticBaselineConfig:
    return StaticBaselineConfig(name="synthetic-baseline", version="1.0-test")


class StaticBaselineTests(TestCase):
    def test_first_matching_rule_wins_and_is_attributed(self) -> None:
        rules = [
            PredicateRule(
                rule_id="failed-escalates",
                provenance=RuleProvenance.SYNTHETIC_ONLY,
                predicate=lambda state: state.evidence.outcome == PhasingOutcome.FAILED,
                decision=StaticRuleDecision(
                    action=DecisionAction.ESCALATE,
                    confidence=1.0,
                    rationale="Synthetic failed outcome.",
                ),
            ),
            PredicateRule(
                rule_id="never-reached",
                provenance=RuleProvenance.SYNTHETIC_ONLY,
                predicate=lambda state: True,
                decision=StaticRuleDecision(
                    action=DecisionAction.ACCEPT,
                    confidence=1.0,
                    rationale="Should not be selected.",
                ),
            ),
        ]
        state = example_state()
        state.evidence = PhasingEvidence(outcome=PhasingOutcome.FAILED)

        result = StaticBaselineAssessor(
            definition=definition(), rules=rules
        ).assess(state)

        self.assertEqual(result.action, DecisionAction.ESCALATE)
        self.assertEqual(result.source, DecisionSource.DETERMINISTIC)
        self.assertIn("failed-escalates", result.rationale)
        self.assertIn("synthetic_only", result.rationale)

    def test_no_match_uses_versioned_fallback(self) -> None:
        result = StaticBaselineAssessor(
            definition=definition(), rules=[]
        ).assess(example_state())

        self.assertEqual(result.action, DecisionAction.ESCALATE)
        self.assertIn("synthetic-baseline@1.0-test", result.rationale)
        self.assertIn("rule=fallback", result.rationale)

    def test_duplicate_rule_ids_are_rejected(self) -> None:
        rule = PredicateRule(
            rule_id="duplicate",
            provenance=RuleProvenance.SYNTHETIC_ONLY,
            predicate=lambda state: True,
            decision=StaticRuleDecision(
                action=DecisionAction.ACCEPT,
                confidence=1.0,
                rationale="Synthetic.",
            ),
        )

        with self.assertRaises(StaticBaselineError):
            StaticBaselineAssessor(definition=definition(), rules=[rule, rule])

    def test_retry_rule_requires_preapproved_exact_strategy(self) -> None:
        proposed = PhasingStrategy(
            strategy_id="retry-bcs",
            parameters={"block_cut_sensitivity": 2},
        )
        rule = PredicateRule(
            rule_id="retry",
            provenance=RuleProvenance.SYNTHETIC_ONLY,
            predicate=lambda state: True,
            decision=StaticRuleDecision(
                action=DecisionAction.RETRY,
                confidence=1.0,
                rationale="Synthetic retry.",
                retry_strategy=proposed,
            ),
        )
        assessor = StaticBaselineAssessor(
            definition=definition(),
            rules=[rule],
            approved_retry_strategies=[],
        )

        with self.assertRaises(StaticBaselineError):
            assessor.assess(example_state())

        approved_assessor = StaticBaselineAssessor(
            definition=definition(),
            rules=[rule],
            approved_retry_strategies=[proposed],
        )
        self.assertEqual(
            approved_assessor.assess(example_state()).retry_strategy,
            proposed,
        )

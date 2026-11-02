"""Safety and fidelity tests for inactive scientific rule candidates."""

from unittest import TestCase

from pydantic import ValidationError

from agent.config import StaticBaselineConfig
from agent.rule_candidates import (
    CANDOTTI_CATEGORY_CANDIDATES,
    CandidateStatus,
    InactiveStaticRuleCandidate,
)
from agent.static_baseline import StaticBaselineAssessor, StaticBaselineError


class CandottiCandidateTests(TestCase):
    def test_all_three_categories_are_present_and_inactive(self) -> None:
        self.assertEqual(
            [candidate.category for candidate in CANDOTTI_CATEGORY_CANDIDATES],
            [1, 2, 3],
        )
        self.assertEqual(
            len({candidate.rule_id for candidate in CANDOTTI_CATEGORY_CANDIDATES}),
            3,
        )
        for candidate in CANDOTTI_CATEGORY_CANDIDATES:
            self.assertEqual(
                candidate.status,
                CandidateStatus.INACTIVE_PENDING_EXPERT_REVIEW,
            )
            self.assertFalse(candidate.runtime_enabled)
            self.assertIsNone(candidate.action)
            self.assertFalse(hasattr(candidate, "evaluate"))

    def test_key_paper_criteria_are_preserved_without_action_mapping(self) -> None:
        category_1, category_2, category_3 = CANDOTTI_CATEGORY_CANDIDATES
        category_1_text = " ".join(
            criterion.source_summary for criterion in category_1.criteria
        )
        category_2_notes = " ".join(
            criterion.interpretation_note or "" for criterion in category_2.criteria
        )
        category_3_text = " ".join(
            criterion.source_summary for criterion in category_3.criteria
        )

        self.assertIn("greater than 80%", category_1_text)
        self.assertIn("one Mendelian error per family", category_1_text)
        self.assertIn("exact 80% boundary", category_2_notes)
        self.assertIn("less than 80%", category_3_text)
        self.assertEqual(category_3.logical_structure, "Any listed criterion is sufficient.")

    def test_candidate_cannot_be_activated_or_assigned_an_action(self) -> None:
        candidate = CANDOTTI_CATEGORY_CANDIDATES[0]
        payload = candidate.model_dump()
        payload["runtime_enabled"] = True
        payload["action"] = "accept"

        with self.assertRaises(ValidationError):
            InactiveStaticRuleCandidate.model_validate(payload)

    def test_static_baseline_rejects_inactive_candidate(self) -> None:
        with self.assertRaisesRegex(StaticBaselineError, "non-executable"):
            StaticBaselineAssessor(
                definition=StaticBaselineConfig(name="review-test", version="1"),
                rules=[CANDOTTI_CATEGORY_CANDIDATES[0]],  # type: ignore[list-item]
            )

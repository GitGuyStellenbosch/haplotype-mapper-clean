"""Inactive scientific rule candidates awaiting expert review.

These records preserve published criteria in a structured, searchable form. They
are deliberately not ``StaticRule`` implementations: they have no ``evaluate``
method, no action, and cannot be enabled at runtime. Candotti's categories describe
target-region performance across families, whereas the agent currently decides one
sample/region case at a time.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.static_baseline import RuleProvenance


CANDOTTI_PAPER_URL = "https://doi.org/10.1111/tpj.16026"


class CandidateStatus(str, Enum):
    """Review state for a proposed scientific rule."""

    INACTIVE_PENDING_EXPERT_REVIEW = "inactive_pending_expert_review"


class CandidateCriterion(BaseModel):
    """One non-executable statement extracted from a source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion_id: str = Field(min_length=1)
    source_summary: str = Field(min_length=1)
    required_evidence: tuple[str, ...] = Field(min_length=1)
    interpretation_note: str | None = None


class InactiveStaticRuleCandidate(BaseModel):
    """A reviewable candidate that is structurally unable to make a decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str = Field(min_length=1)
    category: Literal[1, 2, 3]
    provenance: Literal[RuleProvenance.CANDOTTI] = RuleProvenance.CANDOTTI
    status: Literal[
        CandidateStatus.INACTIVE_PENDING_EXPERT_REVIEW
    ] = CandidateStatus.INACTIVE_PENDING_EXPERT_REVIEW
    runtime_enabled: Literal[False] = False
    action: None = None
    source_url: str = CANDOTTI_PAPER_URL
    source_location: str = "Methods: Haplotype quality control"
    logical_structure: str = Field(min_length=1)
    criteria: tuple[CandidateCriterion, ...] = Field(min_length=1)
    scope_warning: str = (
        "Region-level evidence across full-sib families; not a per-sample action label."
    )


CANDOTTI_CATEGORY_CANDIDATES: tuple[InactiveStaticRuleCandidate, ...] = (
    InactiveStaticRuleCandidate(
        rule_id="candotti-category-1-candidate",
        category=1,
        logical_structure="All listed criteria are required.",
        criteria=(
            CandidateCriterion(
                criterion_id="mendelian-all-families",
                source_summary=(
                    "Passes the Mendelian check in all full-sib families, allowing "
                    "one Mendelian error per family."
                ),
                required_evidence=("per_family_mendelian_errors",),
            ),
            CandidateCriterion(
                criterion_id="parental-haplotypes-one-family",
                source_summary=(
                    "Both parental haplotypes are correctly called in at least one "
                    "full-sib family."
                ),
                required_evidence=("per_family_parental_haplotype_call_status",),
            ),
            CandidateCriterion(
                criterion_id="call-rate-above-80-one-family",
                source_summary=(
                    "Call rate is greater than 80% in at least one full-sib family."
                ),
                required_evidence=("per_family_call_rate",),
                interpretation_note=(
                    "The paper's family-level call rate is not RegionState.phasing "
                    "evidence or WhatsHap phased_fraction."
                ),
            ),
            CandidateCriterion(
                criterion_id="no-unphased-snps",
                source_summary="No unphased SNPs are present.",
                required_evidence=("per_family_unphased_snp_count",),
                interpretation_note=(
                    "The exact aggregation scope should be confirmed before coding "
                    "an executable predicate."
                ),
            ),
        ),
    ),
    InactiveStaticRuleCandidate(
        rule_id="candotti-category-2-candidate",
        category=2,
        logical_structure=(
            "The qualifying-family criteria are required, plus at least one of the "
            "two cross-family exception criteria."
        ),
        criteria=(
            CandidateCriterion(
                criterion_id="mendelian-at-least-one-family",
                source_summary=(
                    "Passes the Mendelian check in at least one full-sib family, "
                    "allowing one Mendelian error per family."
                ),
                required_evidence=("per_family_mendelian_errors",),
            ),
            CandidateCriterion(
                criterion_id="qualifying-family-complete-parents",
                source_summary="Has no missing parent data in the qualifying family.",
                required_evidence=("per_family_parent_data_complete",),
            ),
            CandidateCriterion(
                criterion_id="qualifying-family-80-call-rate",
                source_summary="Has an 80% call rate in the qualifying family.",
                required_evidence=("per_family_call_rate",),
                interpretation_note=(
                    "The publication wording does not resolve how an exact 80% "
                    "boundary should be implemented; expert interpretation is needed."
                ),
            ),
            CandidateCriterion(
                criterion_id="mendelian-failure-some-families",
                source_summary="Fails the Mendelian check in some full-sib families.",
                required_evidence=("per_family_mendelian_errors",),
            ),
            CandidateCriterion(
                criterion_id="unphased-snps-other-families",
                source_summary=(
                    "Has unphased SNPs in some of the other full-sib families."
                ),
                required_evidence=("per_family_unphased_snp_count",),
            ),
        ),
    ),
    InactiveStaticRuleCandidate(
        rule_id="candotti-category-3-candidate",
        category=3,
        logical_structure="Any listed criterion is sufficient.",
        criteria=(
            CandidateCriterion(
                criterion_id="mendelian-check-not-passed",
                source_summary="Does not pass the Mendelian check.",
                required_evidence=("per_family_mendelian_errors",),
                interpretation_note=(
                    "The family aggregation needed for an executable test must be "
                    "confirmed."
                ),
            ),
            CandidateCriterion(
                criterion_id="missing-parental-data",
                source_summary="Has missing parental data.",
                required_evidence=("per_family_parent_data_complete",),
            ),
            CandidateCriterion(
                criterion_id="call-rate-below-80-all-families",
                source_summary=(
                    "Has less than 80% call rate across all full-sib families."
                ),
                required_evidence=("per_family_call_rate",),
            ),
        ),
    ),
)


__all__ = [
    "CANDOTTI_CATEGORY_CANDIDATES",
    "CANDOTTI_PAPER_URL",
    "CandidateCriterion",
    "CandidateStatus",
    "InactiveStaticRuleCandidate",
]

"""LangGraph-compatible nodes for the Haplotype Mapper decision workflow.

These nodes implement orchestration mechanics and safeguards. Assessment is
injected behind a protocol; active research Skills guide model proposals.
"""

from __future__ import annotations

from typing import Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.config import AgentConfig, PipelineMode
from agent.pipeline_interface import PipelineInterface
from agent.state import (
    DecisionAction,
    DecisionRecord,
    DecisionSource,
    PhasingOutcome,
    PhasingStrategy,
    RegionState,
)


Route = Literal["accept", "retry", "escalate"]
NodeUpdate = dict[str, object]
RESEARCH_RETRY_POLICY_VERSIONS = frozenset({
    "research-retries-20261006-v1",  # Historical saved attempts.
    "research-retries-20261006-v2",  # Current phasing-only attempts.
})


class NodeExecutionError(RuntimeError):
    """Raised when a node is invoked in an incompatible state."""


class AssessmentResult(BaseModel):
    """Structured proposal returned by any deterministic, LLM, or human assessor."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: DecisionAction
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    source: DecisionSource
    retry_strategy: PhasingStrategy | None = None

    @model_validator(mode="after")
    def retry_has_exactly_one_strategy(self) -> "AssessmentResult":
        if self.action == DecisionAction.RETRY and self.retry_strategy is None:
            raise ValueError("a retry assessment requires retry_strategy")
        if self.action != DecisionAction.RETRY and self.retry_strategy is not None:
            raise ValueError("retry_strategy is only valid for a retry assessment")
        return self


class Assessor(Protocol):
    """Replaceable source of structured triage proposals."""

    def assess(self, state: RegionState) -> AssessmentResult:
        """Assess one case without mutating its state."""


def _decision_from_assessment(assessment: AssessmentResult) -> DecisionRecord:
    return DecisionRecord(
        action=assessment.action,
        confidence=assessment.confidence,
        rationale=assessment.rationale,
        source=assessment.source,
    )


def _deterministic_escalation(
    state: RegionState,
    warnings: list[str],
    rationale: str,
    proposal: DecisionRecord | None = None,
) -> NodeUpdate:
    """Build the common fail-closed update used by assessment safeguards."""

    decision = DecisionRecord(
        action=DecisionAction.ESCALATE,
        confidence=1.0,
        rationale=rationale,
        source=DecisionSource.DETERMINISTIC,
    )
    update: NodeUpdate = {
        "decision": decision,
        "decision_history": [*state.decision_history, decision],
        "pending_strategy": None,
        "warnings": [*warnings, rationale],
    }
    if proposal is not None:
        update["proposal_history"] = [*state.proposal_history, proposal]
    return update


ROUTINE_TOOL_MESSAGES = (
    "Evidence extracted from an existing WhatsHap-phased VCF",
    "Evidence from an active research retry",
)
ROUTINE_WARNINGS = (
    "Real biological evidence without sample-level ground truth",
    "No pedigree data are available",
)


def acceptance_guard_reason(state: RegionState) -> str | None:
    """Explain the first unmet requirement of the active real-evidence accept rule."""

    evidence = state.evidence
    het = evidence.heterozygous_snp_count
    if evidence.outcome != PhasingOutcome.PHASED:
        return f"outcome is {evidence.outcome.value}, not phased"
    if het is None or het < 2:
        return f"heterozygous SNP count is {het}, not at least two"
    if evidence.phased_snp_count != het:
        return f"phased SNP count {evidence.phased_snp_count} differs from heterozygous count {het}"
    if evidence.unphased_snp_count != 0:
        return f"unphased SNP count is {evidence.unphased_snp_count}, not zero"
    if evidence.call_rate != 1.0:
        return f"call rate is {evidence.call_rate}, not 1.0"
    phase_sets = evidence.additional_metrics.get("phase_set_count")
    if type(phase_sets) is not int or phase_sets != 1:
        return f"regional phase-set count is {phase_sets}, not exactly one"
    if (evidence.additional_metrics.get("retry_genotype_change_count") or 0) > 0:
        return "retry changed genotype calls"
    for message in evidence.tool_messages:
        if not message.startswith(ROUTINE_TOOL_MESSAGES):
            return f"unresolved tool message: {message}"
    for warning in state.warnings:
        if not warning.startswith(ROUTINE_WARNINGS):
            return f"unresolved warning: {warning}"
    return None


def guard_acceptance(
    state: RegionState, assessment: AssessmentResult, config: AgentConfig
) -> AssessmentResult:
    """Fail closed on explicit real-policy accept requirements, preserving proposals."""

    if (config.pipeline.mode != PipelineMode.REAL
            or state.metadata.get("data_status") != "real_evidence_no_ground_truth"
            or assessment.action != DecisionAction.ACCEPT):
        return assessment
    reason = acceptance_guard_reason(state)
    if reason is None:
        return assessment
    return AssessmentResult(
        action=DecisionAction.ESCALATE, confidence=1.0,
        rationale=f"acceptance guard (research policy, unvalidated): {reason}",
        source=DecisionSource.DETERMINISTIC,
    )


def _clear_research_phasing_rule(
    state: RegionState, config: AgentConfig
) -> AssessmentResult | None:
    """Apply the narrow, auditable retry/accept rule on real extracted evidence."""

    if (
        config.pipeline.mode != PipelineMode.REAL
        or state.metadata.get("data_status") != "real_evidence_no_ground_truth"
    ):
        return None
    if any(not message.startswith(ROUTINE_TOOL_MESSAGES) for message in state.evidence.tool_messages):
        return None
    if any(not warning.startswith(ROUTINE_WARNINGS) for warning in state.warnings):
        return None

    evidence = state.evidence
    het = evidence.heterozygous_snp_count
    phase_sets = evidence.additional_metrics.get("phase_set_count")
    if not (
        evidence.outcome == PhasingOutcome.PHASED
        and het is not None and het >= 2
        and evidence.phased_snp_count == het
        and evidence.unphased_snp_count == 0
        and evidence.call_rate == 1.0
        and type(phase_sets) is int
    ):
        return None

    if state.retries_used == 0 and phase_sets > 1:
        strategy = next((item for item in config.pipeline.retry_strategies
                         if item.strategy_id == "lower-block-cut-sensitivity-v1"
                         and item.parameters == {"block_cut_sensitivity": 2}
                         and type(item.parameters["block_cut_sensitivity"]) is int), None)
        if strategy is not None:
            return AssessmentResult(
                action=DecisionAction.RETRY,
                confidence=1.0,
                rationale="phase15-deterministic-policy-v1 (unvalidated): complete genotype calls and multiple regional phase sets trigger one sensitivity-2 diagnostic retry.",
                source=DecisionSource.DETERMINISTIC,
                retry_strategy=strategy,
            )
    if (
        state.retries_used > 0
        and state.attempt_history[-1].strategy_id == "lower-block-cut-sensitivity-v1"
        and phase_sets == 1
        and evidence.additional_metrics.get("retry_genotype_change_count") == 0
        and evidence.additional_metrics.get("retry_policy_version") in RESEARCH_RETRY_POLICY_VERSIONS
    ):
        return AssessmentResult(
            action=DecisionAction.ACCEPT,
            confidence=1.0,
            rationale="phase15-deterministic-policy-v1 (unvalidated): all observed heterozygous SNPs are phased in one regional phase set with complete genotype calls.",
            source=DecisionSource.DETERMINISTIC,
        )
    return None


def make_assess_node(
    assessor: Assessor,
    config: AgentConfig,
) -> Callable[[RegionState], NodeUpdate]:
    """Build an assessment node with deterministic failure-awareness safeguards."""

    def assess_node(state: RegionState) -> NodeUpdate:
        # Research retry stopping rules use observed evidence, before inference.
        if state.evidence.additional_metrics.get("retry_policy_version") in RESEARCH_RETRY_POLICY_VERSIONS:
            warnings = list(state.warnings)
            if state.evidence.additional_metrics.get("retry_genotype_change_count", 0) > 0:
                return _deterministic_escalation(
                    state, warnings, "research retry changed regional genotypes; review the alternative VCF"
                )
            if len(state.attempt_history) >= 2:
                previous = state.attempt_history[-2].evidence
                current = state.evidence
                fields = ("heterozygous_snp_count", "phased_snp_count", "unphased_snp_count", "call_rate")
                values_before = [getattr(previous, field) for field in fields] + [previous.additional_metrics.get("phase_set_count")]
                values_after = [getattr(current, field) for field in fields] + [current.additional_metrics.get("phase_set_count")]
                if all(value is not None for value in values_before + values_after) and values_before == values_after:
                    return _deterministic_escalation(
                        state, warnings, "research retry left regional counts, call rate, and phase sets unchanged"
                    )
        assessment = _clear_research_phasing_rule(state, config) or assessor.assess(state)
        proposal = _decision_from_assessment(assessment)
        warnings = list(state.warnings)

        guarded = guard_acceptance(state, assessment, config)
        if guarded is not assessment:
            return _deterministic_escalation(state, warnings, guarded.rationale, proposal)

        minimum_confidence = config.decisions.minimum_confidence
        if (
            minimum_confidence is not None
            and assessment.action != DecisionAction.ESCALATE
            and assessment.confidence < minimum_confidence
        ):
            warning = (
                "model proposal replaced with escalation because confidence "
                f"{assessment.confidence:.3f} was below the configured minimum "
                f"{minimum_confidence:.3f}"
            )
            return _deterministic_escalation(state, warnings, warning, proposal)

        if assessment.action == DecisionAction.RETRY:
            if state.retries_used >= config.decisions.max_retries:
                warning = (
                    "retry proposal replaced with escalation because the configured "
                    "retry limit was reached"
                )
                return _deterministic_escalation(state, warnings, warning, proposal)

            strategy = assessment.retry_strategy
            if strategy is None:  # Defensive check after Pydantic validation.
                raise NodeExecutionError("retry assessment did not provide a strategy")
            if config.pipeline.mode == PipelineMode.REAL:
                allowed = next((item for item in config.pipeline.retry_strategies if item == strategy), None)
                if allowed is None or any(type(strategy.parameters[k]) is not type(v)
                                          for k, v in allowed.parameters.items()):
                    return _deterministic_escalation(state, warnings, "retry is not an exact allowlisted strategy", proposal)
                if strategy.strategy_id == "lower-block-cut-sensitivity-v1":
                    heterozygous = state.evidence.heterozygous_snp_count
                    phase_sets = state.evidence.additional_metrics.get("phase_set_count")
                    if heterozygous is None or heterozygous < 2 or type(phase_sets) is not int or phase_sets < 2:
                        return _deterministic_escalation(state, warnings, "block-cut retry requires at least two heterozygous SNPs and multiple observed phase sets", proposal)
            if strategy.strategy_id in state.strategies_tried:
                warning = (
                    f"retry proposal replaced with escalation because strategy "
                    f"'{strategy.strategy_id}' was already attempted"
                )
                return _deterministic_escalation(state, warnings, warning, proposal)

        decision = _decision_from_assessment(assessment)
        return {
            "decision": decision,
            "decision_history": [*state.decision_history, decision],
            "proposal_history": [*state.proposal_history, proposal],
            "pending_strategy": assessment.retry_strategy,
            "warnings": warnings,
        }

    return assess_node


def make_retry_node(
    pipeline: PipelineInterface,
) -> Callable[[RegionState], NodeUpdate]:
    """Build a retry node that runs exactly one pending deterministic strategy."""

    def retry_node(state: RegionState) -> NodeUpdate:
        if state.decision is None or state.decision.action != DecisionAction.RETRY:
            raise NodeExecutionError("retry node requires a retry decision")
        if state.pending_strategy is None:
            raise NodeExecutionError("retry node requires a pending strategy")

        attempt = pipeline.run(state, state.pending_strategy)
        return {
            "attempt_history": [*state.attempt_history, attempt],
            "evidence": attempt.evidence,
            "decision": None,
            "pending_strategy": None,
        }

    return retry_node


def accept_node(state: RegionState) -> NodeUpdate:
    """Validate entry into the successful terminal branch."""

    if state.decision is None or state.decision.action != DecisionAction.ACCEPT:
        raise NodeExecutionError("accept node requires an accept decision")
    return {}


def escalate_node(state: RegionState) -> NodeUpdate:
    """Validate entry into the expert-review terminal branch."""

    if state.decision is None or state.decision.action != DecisionAction.ESCALATE:
        raise NodeExecutionError("escalate node requires an escalate decision")
    return {}


def route_decision(state: RegionState) -> Route:
    """Convert the current validated decision into a LangGraph route name."""

    if state.decision is None:
        raise NodeExecutionError("cannot route a state without a decision")
    return state.decision.action.value


__all__ = [
    "AssessmentResult",
    "Assessor",
    "NodeExecutionError",
    "acceptance_guard_reason",
    "accept_node",
    "escalate_node",
    "guard_acceptance",
    "make_assess_node",
    "make_retry_node",
    "route_decision",
]

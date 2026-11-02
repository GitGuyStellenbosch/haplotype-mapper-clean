"""Versioned fixed-rule comparator for the static side of the experiment.

This module defines how deterministic rules are ordered, attributed, and audited.
It deliberately does not define biological rules before they are justified.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.config import StaticBaselineConfig
from agent.nodes import AssessmentResult
from agent.state import (
    DecisionAction,
    DecisionSource,
    PhasingStrategy,
    RegionState,
)


class StaticBaselineError(ValueError):
    """Raised when a static rule set is unsafe or internally inconsistent."""


class RuleProvenance(str, Enum):
    """Evidence class supporting a static decision rule."""

    PRIMARY_DOCUMENTATION = "primary_documentation"
    CANDOTTI = "candotti_et_al"
    PROJECT_DEFINITION = "project_definition"
    EXPERT_VALIDATED = "expert_validated"
    SYNTHETIC_ONLY = "synthetic_only"


class StaticRuleDecision(BaseModel):
    """Decision emitted when a static rule matches one state."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: DecisionAction
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    retry_strategy: PhasingStrategy | None = None

    @model_validator(mode="after")
    def retry_requires_strategy(self) -> "StaticRuleDecision":
        if self.action == DecisionAction.RETRY and self.retry_strategy is None:
            raise ValueError("a static retry rule requires retry_strategy")
        if self.action != DecisionAction.RETRY and self.retry_strategy is not None:
            raise ValueError("retry_strategy is only valid for retry rules")
        return self


class StaticRule(Protocol):
    """Ordered deterministic rule evaluated by the static baseline."""

    rule_id: str
    provenance: RuleProvenance

    def evaluate(self, state: RegionState) -> StaticRuleDecision | None:
        """Return a decision when matched, otherwise defer to the next rule."""


@dataclass(frozen=True)
class PredicateRule:
    """Small reusable rule implementation, primarily useful for declared rules."""

    rule_id: str
    provenance: RuleProvenance
    predicate: Callable[[RegionState], bool]
    decision: StaticRuleDecision

    def evaluate(self, state: RegionState) -> StaticRuleDecision | None:
        return self.decision if self.predicate(state) else None


class StaticBaselineAssessor:
    """Evaluate ordered deterministic rules and emit one auditable assessment."""

    def __init__(
        self,
        *,
        definition: StaticBaselineConfig,
        rules: Sequence[StaticRule],
        approved_retry_strategies: Sequence[PhasingStrategy] = (),
    ) -> None:
        self.definition = definition
        self.rules = tuple(rules)
        non_executable = [
            rule.rule_id
            for rule in self.rules
            if not callable(getattr(rule, "evaluate", None))
        ]
        if non_executable:
            raise StaticBaselineError(
                "static baseline received non-executable rule candidates: "
                + ", ".join(non_executable)
            )
        rule_ids = [rule.rule_id for rule in self.rules]
        if any(not rule_id.strip() for rule_id in rule_ids):
            raise StaticBaselineError("static rule IDs cannot be blank")
        if len(rule_ids) != len(set(rule_ids)):
            raise StaticBaselineError("static rule IDs must be unique")
        self._approved_strategies = {
            strategy.strategy_id: strategy
            for strategy in approved_retry_strategies
        }
        if len(self._approved_strategies) != len(approved_retry_strategies):
            raise StaticBaselineError("approved retry strategy IDs must be unique")

    def _validate_retry(self, decision: StaticRuleDecision) -> None:
        if decision.retry_strategy is None:
            return
        proposed = decision.retry_strategy
        approved = self._approved_strategies.get(proposed.strategy_id)
        if approved is None:
            raise StaticBaselineError(
                f"static rule proposed unapproved strategy '{proposed.strategy_id}'"
            )
        if proposed != approved:
            raise StaticBaselineError(
                f"static rule altered approved strategy '{proposed.strategy_id}'"
            )

    def assess(self, state: RegionState) -> AssessmentResult:
        for rule in self.rules:
            decision = rule.evaluate(state)
            if decision is None:
                continue
            self._validate_retry(decision)
            prefix = (
                f"baseline={self.definition.name}@{self.definition.version}; "
                f"rule={rule.rule_id}; provenance={rule.provenance.value}"
            )
            return AssessmentResult(
                action=decision.action,
                confidence=decision.confidence,
                rationale=f"[{prefix}] {decision.rationale}",
                source=DecisionSource.DETERMINISTIC,
                retry_strategy=decision.retry_strategy,
            )

        fallback = self.definition.fallback_action
        return AssessmentResult(
            action=fallback,
            confidence=self.definition.fallback_confidence,
            rationale=(
                f"[baseline={self.definition.name}@{self.definition.version}; "
                "rule=fallback; provenance=project_definition] "
                "No declared static rule matched."
            ),
            source=DecisionSource.DETERMINISTIC,
        )


__all__ = [
    "PredicateRule",
    "RuleProvenance",
    "StaticBaselineAssessor",
    "StaticBaselineError",
    "StaticRule",
    "StaticRuleDecision",
]

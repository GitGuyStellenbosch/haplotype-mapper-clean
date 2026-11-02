"""Unit tests for graph-node orchestration and safeguards."""

from unittest import TestCase

from pydantic import ValidationError

from agent.config import AgentConfig, PipelineMode, load_config
from agent.nodes import (
    AssessmentResult,
    NodeExecutionError,
    accept_node,
    escalate_node,
    make_assess_node,
    make_retry_node,
    route_decision,
)
from agent.pipeline_interface import MockPhasingBackend, PipelineInterface
from agent.state import (
    DecisionAction,
    DecisionRecord,
    DecisionSource,
    PhasingAttempt,
    PhasingEvidence,
    PhasingOutcome,
    PhasingStrategy,
    RegionIdentity,
    RegionState,
)


class ScriptedAssessor:
    """Synthetic assessor that returns one predefined result."""

    def __init__(self, result: AssessmentResult) -> None:
        self.result = result

    def assess(self, state: RegionState) -> AssessmentResult:
        return self.result


def example_config(
    max_retries: int = 2,
    minimum_confidence: float | None = None,
) -> AgentConfig:
    config = load_config()
    decisions = config.decisions.model_copy(
        update={
            "max_retries": max_retries,
            "minimum_confidence": minimum_confidence,
        }
    )
    return config.model_copy(
        update={
            "agent": config.agent.model_copy(update={"decisions": decisions})
        }
    )


def example_state(*, attempt_count: int = 0) -> RegionState:
    attempts = [
        PhasingAttempt(
            attempt_number=index,
            strategy_id=f"strategy-{index}",
            evidence=PhasingEvidence(outcome=PhasingOutcome.FAILED),
        )
        for index in range(1, attempt_count + 1)
    ]
    return RegionState(
        sample_id="synthetic-sample",
        region=RegionIdentity(
            gene="Eucgr.A00001",
            target_region="Gene start region",
            chromosome="Chr01",
            start_position=100,
            end_position=120,
        ),
        attempt_history=attempts,
        evidence=attempts[-1].evidence if attempts else PhasingEvidence(),
    )


def state_with_update(state: RegionState, update: dict[str, object]) -> RegionState:
    return RegionState.model_validate(
        {**state.model_dump(exclude_computed_fields=True), **update}
    )


class NodeTests(TestCase):
    def test_assessment_result_requires_strategy_for_retry(self) -> None:
        with self.assertRaises(ValidationError):
            AssessmentResult(
                action=DecisionAction.RETRY,
                confidence=0.6,
                rationale="Synthetic retry proposal.",
                source=DecisionSource.LLM,
            )

    def test_assess_node_records_accept_decision(self) -> None:
        assessor = ScriptedAssessor(
            AssessmentResult(
                action=DecisionAction.ACCEPT,
                confidence=0.9,
                rationale="Synthetic clean result.",
                source=DecisionSource.LLM,
            )
        )

        update = make_assess_node(assessor, example_config())(example_state())
        updated_state = state_with_update(example_state(), update)

        self.assertEqual(updated_state.decision.action, DecisionAction.ACCEPT)
        self.assertEqual(route_decision(updated_state), "accept")
        self.assertEqual(updated_state.proposal_history[0].action, DecisionAction.ACCEPT)

    def test_real_acceptance_guard_preserves_conflicting_model_proposal(self) -> None:
        config = example_config().with_pipeline_mode(PipelineMode.REAL)
        config.pipeline.retry_strategies = []
        state = example_state()
        state.metadata["data_status"] = "real_evidence_no_ground_truth"
        state.evidence = PhasingEvidence(
            outcome=PhasingOutcome.PHASED, heterozygous_snp_count=3,
            phased_snp_count=3, unphased_snp_count=0, call_rate=1,
            additional_metrics={"phase_set_count": 2},
        )
        state.attempt_history = [PhasingAttempt(
            attempt_number=1, strategy_id="baseline", evidence=state.evidence,
        )]
        assessor = ScriptedAssessor(AssessmentResult(
            action=DecisionAction.ACCEPT, confidence=0.9,
            rationale="Model overlooked the second phase set.", source=DecisionSource.LLM,
        ))

        final = state_with_update(state, make_assess_node(assessor, config)(state))

        self.assertEqual(final.proposal_history[-1].action, DecisionAction.ACCEPT)
        self.assertEqual(final.proposal_history[-1].source, DecisionSource.LLM)
        self.assertEqual(final.decision.action, DecisionAction.ESCALATE)
        self.assertEqual(final.decision.source, DecisionSource.DETERMINISTIC)
        self.assertIn("phase-set count is 2", final.decision.rationale)

        state.evidence.additional_metrics["phase_set_count"] = 1
        state.attempt_history[0].evidence.additional_metrics["phase_set_count"] = 1
        accepted = state_with_update(state, make_assess_node(assessor, config)(state))
        self.assertEqual(accepted.decision.action, DecisionAction.ACCEPT)
        self.assertEqual(accepted.decision.source, DecisionSource.LLM)

        state.evidence.tool_messages.append("Unresolved PS-format problem")
        warned = state_with_update(state, make_assess_node(assessor, config)(state))
        self.assertEqual(warned.decision.action, DecisionAction.ESCALATE)
        self.assertIn("Unresolved PS-format problem", warned.decision.rationale)

    def test_assess_node_records_pending_retry_strategy(self) -> None:
        strategy = PhasingStrategy(strategy_id="retry-new")
        assessor = ScriptedAssessor(
            AssessmentResult(
                action=DecisionAction.RETRY,
                confidence=0.7,
                rationale="Synthetic recoverable failure.",
                source=DecisionSource.LLM,
                retry_strategy=strategy,
            )
        )

        update = make_assess_node(assessor, example_config())(example_state())

        self.assertEqual(update["pending_strategy"], strategy)

    def test_low_confidence_proposal_is_deterministically_escalated(self) -> None:
        assessor = ScriptedAssessor(
            AssessmentResult(
                action=DecisionAction.ACCEPT,
                confidence=0.6,
                rationale="Synthetic low-confidence acceptance.",
                source=DecisionSource.LLM,
            )
        )

        update = make_assess_node(
            assessor,
            example_config(minimum_confidence=0.7),
        )(example_state())
        updated_state = state_with_update(example_state(), update)

        self.assertEqual(updated_state.decision.action, DecisionAction.ESCALATE)
        self.assertEqual(updated_state.decision.source, DecisionSource.DETERMINISTIC)
        self.assertIn("below the configured minimum", updated_state.warnings[-1])

    def test_retry_limit_converts_retry_to_escalation(self) -> None:
        assessor = ScriptedAssessor(
            AssessmentResult(
                action=DecisionAction.RETRY,
                confidence=0.7,
                rationale="Synthetic retry proposal.",
                source=DecisionSource.LLM,
                retry_strategy=PhasingStrategy(strategy_id="retry-3"),
            )
        )
        state = example_state(attempt_count=3)

        update = make_assess_node(assessor, example_config(max_retries=2))(state)
        updated_state = state_with_update(state, update)

        self.assertEqual(updated_state.decision.action, DecisionAction.ESCALATE)
        self.assertEqual(updated_state.decision.source, DecisionSource.DETERMINISTIC)

    def test_repeated_retry_strategy_converts_to_escalation(self) -> None:
        state = example_state(attempt_count=1)
        assessor = ScriptedAssessor(
            AssessmentResult(
                action=DecisionAction.RETRY,
                confidence=0.7,
                rationale="Synthetic repeated strategy.",
                source=DecisionSource.LLM,
                retry_strategy=PhasingStrategy(strategy_id="strategy-1"),
            )
        )

        update = make_assess_node(assessor, example_config())(state)
        updated_state = state_with_update(state, update)

        self.assertEqual(updated_state.decision.action, DecisionAction.ESCALATE)
        self.assertTrue(updated_state.warnings)

    def test_retry_node_runs_pipeline_and_clears_proposal(self) -> None:
        state = example_state(attempt_count=1)
        state = state_with_update(
            state,
            {
                "decision": DecisionRecord(
                    action=DecisionAction.RETRY,
                    confidence=0.7,
                    rationale="Synthetic retry.",
                    source=DecisionSource.LLM,
                ),
                "pending_strategy": PhasingStrategy(strategy_id="strategy-2"),
            },
        )
        pipeline = PipelineInterface(
            MockPhasingBackend(
                {
                    "strategy-2": PhasingEvidence(
                        outcome=PhasingOutcome.PHASED,
                        haplotype_count=2,
                    )
                }
            ),
            validate_inputs=False,
        )

        update = make_retry_node(pipeline)(state)
        updated_state = state_with_update(state, update)

        self.assertEqual(len(updated_state.attempt_history), 2)
        self.assertEqual(updated_state.evidence.outcome, PhasingOutcome.PHASED)
        self.assertIsNone(updated_state.decision)
        self.assertIsNone(updated_state.pending_strategy)

    def test_terminal_nodes_reject_wrong_decision(self) -> None:
        state = example_state()

        with self.assertRaises(NodeExecutionError):
            accept_node(state)
        with self.assertRaises(NodeExecutionError):
            escalate_node(state)

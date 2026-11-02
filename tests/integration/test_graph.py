"""Synthetic end-to-end tests for the compiled LangGraph workflow."""

from unittest import TestCase

from agent.config import AgentConfig, load_config
from agent.graph import GraphInputError, build_agent_graph
from agent.nodes import AssessmentResult
from agent.pipeline_interface import MockPhasingBackend, PipelineInterface
from agent.state import (
    DecisionAction,
    DecisionSource,
    PhasingAttempt,
    PhasingEvidence,
    PhasingOutcome,
    PhasingStrategy,
    RegionIdentity,
    RegionState,
)


class SequenceAssessor:
    """Return a fixed sequence of explicitly synthetic assessments."""

    def __init__(self, assessments: list[AssessmentResult]) -> None:
        self._assessments = iter(assessments)
        self.calls = 0

    def assess(self, state: RegionState) -> AssessmentResult:
        self.calls += 1
        return next(self._assessments)


class AlwaysRetryWithNewStrategy:
    """Synthetic assessor used to exercise the deterministic retry limit."""

    def assess(self, state: RegionState) -> AssessmentResult:
        next_number = len(state.attempt_history) + 1
        return AssessmentResult(
            action=DecisionAction.RETRY,
            confidence=0.5,
            rationale="Synthetic retry-limit scenario.",
            source=DecisionSource.LLM,
            retry_strategy=PhasingStrategy(
                strategy_id=f"strategy-{next_number}"
            ),
        )


def example_config(max_retries: int = 2) -> AgentConfig:
    config = load_config()
    decisions = config.decisions.model_copy(update={"max_retries": max_retries})
    return config.model_copy(
        update={
            "agent": config.agent.model_copy(update={"decisions": decisions})
        }
    )


def initial_state() -> RegionState:
    """Create a case whose initial deterministic phasing run has failed."""

    baseline = PhasingAttempt(
        attempt_number=1,
        strategy_id="strategy-1",
        evidence=PhasingEvidence(outcome=PhasingOutcome.FAILED),
    )
    return RegionState(
        sample_id="synthetic-sample",
        region=RegionIdentity(
            gene="Eucgr.A00001",
            target_region="Gene start region",
            chromosome="Chr01",
            start_position=100,
            end_position=120,
        ),
        evidence=baseline.evidence,
        attempt_history=[baseline],
    )


def run_graph(
    assessor: object,
    backend: MockPhasingBackend,
    *,
    max_retries: int = 2,
) -> dict[str, object]:
    graph = build_agent_graph(
        config=example_config(max_retries),
        assessor=assessor,
        pipeline=PipelineInterface(backend, validate_inputs=False),
    )
    return graph.invoke(initial_state())


class GraphIntegrationTests(TestCase):
    def test_case_without_baseline_attempt_is_rejected(self) -> None:
        assessor = SequenceAssessor(
            [
                AssessmentResult(
                    action=DecisionAction.ACCEPT,
                    confidence=0.9,
                    rationale="Should not be reached.",
                    source=DecisionSource.DETERMINISTIC,
                )
            ]
        )
        graph = build_agent_graph(
            config=example_config(),
            assessor=assessor,
            pipeline=PipelineInterface(
                MockPhasingBackend({}), validate_inputs=False
            ),
        )
        incomplete_state = initial_state().model_copy(
            update={"attempt_history": []}
        )

        with self.assertRaises(GraphInputError):
            graph.invoke(incomplete_state)

        self.assertEqual(assessor.calls, 0)

    def test_immediate_acceptance_reaches_terminal_state(self) -> None:
        assessor = SequenceAssessor(
            [
                AssessmentResult(
                    action=DecisionAction.ACCEPT,
                    confidence=0.9,
                    rationale="Synthetic acceptance.",
                    source=DecisionSource.DETERMINISTIC,
                )
            ]
        )

        result = run_graph(assessor, MockPhasingBackend({}))

        self.assertEqual(result["decision"].action, DecisionAction.ACCEPT)
        self.assertEqual(assessor.calls, 1)

    def test_immediate_escalation_reaches_terminal_state(self) -> None:
        assessor = SequenceAssessor(
            [
                AssessmentResult(
                    action=DecisionAction.ESCALATE,
                    confidence=0.8,
                    rationale="Synthetic escalation.",
                    source=DecisionSource.DETERMINISTIC,
                )
            ]
        )

        result = run_graph(assessor, MockPhasingBackend({}))

        self.assertEqual(result["decision"].action, DecisionAction.ESCALATE)

    def test_retry_loops_back_to_assessment_then_accepts(self) -> None:
        assessor = SequenceAssessor(
            [
                AssessmentResult(
                    action=DecisionAction.RETRY,
                    confidence=0.6,
                    rationale="Synthetic recoverable failure.",
                    source=DecisionSource.LLM,
                    retry_strategy=PhasingStrategy(strategy_id="strategy-2"),
                ),
                AssessmentResult(
                    action=DecisionAction.ACCEPT,
                    confidence=0.9,
                    rationale="Synthetic retry succeeded.",
                    source=DecisionSource.LLM,
                ),
            ]
        )
        backend = MockPhasingBackend(
            {
                "strategy-2": PhasingEvidence(
                    outcome=PhasingOutcome.PHASED,
                    haplotype_count=2,
                )
            }
        )

        result = run_graph(assessor, backend)

        self.assertEqual(result["decision"].action, DecisionAction.ACCEPT)
        self.assertEqual(len(result["attempt_history"]), 2)
        self.assertEqual(result["evidence"].outcome, PhasingOutcome.PHASED)
        self.assertEqual(assessor.calls, 2)
        self.assertEqual(len(backend.requests), 1)

    def test_retry_exhaustion_is_forced_to_escalation(self) -> None:
        backend = MockPhasingBackend(
            {
                "strategy-2": PhasingEvidence(outcome=PhasingOutcome.FAILED),
            }
        )

        result = run_graph(
            AlwaysRetryWithNewStrategy(),
            backend,
            max_retries=1,
        )

        self.assertEqual(result["decision"].action, DecisionAction.ESCALATE)
        self.assertEqual(result["decision"].source, DecisionSource.DETERMINISTIC)
        self.assertEqual(len(result["attempt_history"]), 2)
        self.assertTrue(result["warnings"])

    def test_repeated_strategy_is_forced_to_escalation_without_running(self) -> None:
        assessor = SequenceAssessor(
            [
                AssessmentResult(
                    action=DecisionAction.RETRY,
                    confidence=0.5,
                    rationale="Synthetic repeated strategy.",
                    source=DecisionSource.LLM,
                    retry_strategy=PhasingStrategy(strategy_id="strategy-1"),
                )
            ]
        )
        backend = MockPhasingBackend(
            {"strategy-1": PhasingEvidence(outcome=PhasingOutcome.PHASED)}
        )

        result = run_graph(assessor, backend)

        self.assertEqual(result["decision"].action, DecisionAction.ESCALATE)
        self.assertEqual(backend.requests, [])

"""Construction of the Haplotype Mapper LangGraph workflow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.config import AgentConfig
from agent.nodes import (
    Assessor,
    accept_node,
    escalate_node,
    make_assess_node,
    make_retry_node,
    route_decision,
)
from agent.pipeline_interface import PipelineInterface
from agent.state import RegionState


class GraphInputError(ValueError):
    """Raised when a case lacks the baseline evidence required for assessment."""


def _validate_entry_state(state: RegionState) -> dict[str, object]:
    """Require an auditable baseline attempt before agent triage begins."""

    if not state.attempt_history:
        raise GraphInputError(
            "agent triage requires a recorded baseline phasing attempt"
        )
    if state.evidence != state.attempt_history[-1].evidence:
        raise GraphInputError(
            "current evidence must match the most recent phasing attempt"
        )
    return {}


def build_agent_graph(
    *,
    config: AgentConfig,
    assessor: Assessor,
    pipeline: PipelineInterface,
) -> CompiledStateGraph:
    """Build and compile the accept/retry/escalate workflow.

    Dependencies are injected so the same graph can use scripted assessments during
    preparation and model-backed assessments once domain guidance is validated.
    """

    workflow = StateGraph(RegionState)
    workflow.add_node("validate_entry", _validate_entry_state)
    workflow.add_node("assess", make_assess_node(assessor, config))
    workflow.add_node("retry", make_retry_node(pipeline))
    workflow.add_node("accept", accept_node)
    workflow.add_node("escalate", escalate_node)

    workflow.add_edge(START, "validate_entry")
    workflow.add_edge("validate_entry", "assess")
    workflow.add_conditional_edges(
        "assess",
        route_decision,
        {
            "accept": "accept",
            "retry": "retry",
            "escalate": "escalate",
        },
    )
    workflow.add_edge("retry", "assess")
    workflow.add_edge("accept", END)
    workflow.add_edge("escalate", END)

    return workflow.compile()


__all__ = ["GraphInputError", "build_agent_graph"]

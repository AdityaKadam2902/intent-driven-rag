"""
Builds the query-time LangGraph pipeline.

Per Section 4 of the architecture docs: this is an explicit pipeline,
not an autonomous agent. Every node and edge here is code we control -
the LLM is called INSIDE nodes (classify_intention, resolve_follow_up,
synthesize) to produce specific outputs, but the routing decisions
between nodes are deterministic Python code (route_after_intention
below), not something the model decides.

Current state: the full path now exists for all six intentions.
informational goes straight to retrieve; follow_up first resolves the
question against conversation history (resolve_follow_up), then joins
the same retrieve -> rerank -> synthesize path. Reranking is currently
a simple top-N slice (see rerank.py's docstring for why, and what a
future upgrade looks like).
"""
from langgraph.graph import END, START, StateGraph

from app.graph.nodes.generation.synthesize import synthesize
from app.graph.nodes.retrieval.rerank import rerank
from app.graph.nodes.retrieval.resolve_follow_up import resolve_follow_up
from app.graph.nodes.retrieval.retrieve import retrieve
from app.graph.nodes.routing.classify_intention import classify_intention
from app.graph.nodes.routing.terminal_responses import (
    answer_actionable,
    answer_chitchat,
    answer_meta,
    route_to_escalation,
)
from app.graph.state import GraphState


def route_after_intention(state: GraphState) -> str:
    """The router function LangGraph calls after classify_intention
    completes. Per LangGraph convention, this ONLY reads state and
    returns a string - no LLM calls, no side effects, all real
    computation already happened in the classify_intention node."""
    intention = state["intention"]

    if intention == "follow_up":
        return "follow_up"
    if intention == "informational":
        return "informational"
    return intention  # chitchat, meta, actionable, escalation


def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("classify_intention", classify_intention)
    builder.add_node("answer_chitchat", answer_chitchat)
    builder.add_node("answer_meta", answer_meta)
    builder.add_node("answer_actionable", answer_actionable)
    builder.add_node("route_to_escalation", route_to_escalation)
    builder.add_node("resolve_follow_up", resolve_follow_up)
    builder.add_node("retrieve", retrieve)
    builder.add_node("rerank", rerank)
    builder.add_node("synthesize", synthesize)

    builder.add_edge(START, "classify_intention")

    builder.add_conditional_edges(
        "classify_intention",
        route_after_intention,
        {
            "chitchat": "answer_chitchat",
            "meta": "answer_meta",
            "actionable": "answer_actionable",
            "escalation": "route_to_escalation",
            "informational": "retrieve",
            "follow_up": "resolve_follow_up",
        },
    )

    builder.add_edge("answer_chitchat", END)
    builder.add_edge("answer_meta", END)
    builder.add_edge("answer_actionable", END)
    builder.add_edge("route_to_escalation", END)

    builder.add_edge("resolve_follow_up", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()

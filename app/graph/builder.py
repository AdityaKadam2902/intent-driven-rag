"""
Builds the query-time LangGraph pipeline.

Cache is checked right after classify_intention, before retrieve - a
hit routes straight to END, skipping retrieve/rerank/synthesize
entirely. A miss falls through to the normal retrieval path, with
cache_write running after synthesize to populate the cache for next
time.
"""
from langgraph.graph import END, START, StateGraph

from app.graph.nodes.cache.cache_lookup import cache_lookup
from app.graph.nodes.cache.cache_write import cache_write
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
    intention = state["intention"]
    if intention == "follow_up":
        return "follow_up"
    if intention == "informational":
        return "informational"
    return intention


def route_after_cache_lookup(state: GraphState) -> str:
    """Hit -> straight to END (the cached answer/confidence/sources
    are already in state from cache_lookup). Miss -> continue to
    retrieve as normal."""
    return "hit" if state.get("cache_hit") else "miss"


def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("classify_intention", classify_intention)
    builder.add_node("answer_chitchat", answer_chitchat)
    builder.add_node("answer_meta", answer_meta)
    builder.add_node("answer_actionable", answer_actionable)
    builder.add_node("route_to_escalation", route_to_escalation)
    builder.add_node("resolve_follow_up", resolve_follow_up)
    builder.add_node("cache_lookup", cache_lookup)
    builder.add_node("retrieve", retrieve)
    builder.add_node("rerank", rerank)
    builder.add_node("synthesize", synthesize)
    builder.add_node("cache_write", cache_write)

    builder.add_edge(START, "classify_intention")

    builder.add_conditional_edges(
        "classify_intention",
        route_after_intention,
        {
            "chitchat": "answer_chitchat",
            "meta": "answer_meta",
            "actionable": "answer_actionable",
            "escalation": "route_to_escalation",
            "informational": "cache_lookup",
            "follow_up": "resolve_follow_up",
        },
    )

    builder.add_edge("answer_chitchat", END)
    builder.add_edge("answer_meta", END)
    builder.add_edge("answer_actionable", END)
    builder.add_edge("route_to_escalation", END)

    builder.add_edge("resolve_follow_up", "cache_lookup")

    builder.add_conditional_edges(
        "cache_lookup",
        route_after_cache_lookup,
        {"hit": END, "miss": "retrieve"},
    )

    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "synthesize")
    builder.add_edge("synthesize", "cache_write")
    builder.add_edge("cache_write", END)

    return builder.compile()

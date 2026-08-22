"""
Retrieve node - wraps the already-built, already-tested hybrid_retriever
as a LangGraph node. Nothing about hybrid search itself is reimplemented
here; this is purely the adapter between "graph state in" and "graph
state out" so the retriever can sit in the pipeline.

Per Section 4's design: top 20 retrieved here, narrowed to top 5 by the
rerank node that follows.
"""
from app.graph.state import GraphState
from app.retrieval.retrievers.hybrid_retriever import hybrid_search

RETRIEVE_TOP_N = 20


def retrieve(state: GraphState) -> dict:
    """Uses the resolved query if a prior node (resolve_follow_up) set
    one, otherwise falls back to the raw user_message directly - this
    is what lets informational and follow_up share this same node."""
    query = state.get("resolved_query") or state["user_message"]

    results = hybrid_search(query, limit=RETRIEVE_TOP_N)

    return {"retrieved_chunks": results}

"""
Retrieve node - wraps hybrid_retriever as a LangGraph node.

Postgres circuit breaker: if hybrid_search() raises CircuitOpenError
(the DB breaker is open, see app/database/postgres.py), this node
catches it and sets retrieved_chunks to an empty list plus a
db_unavailable flag - synthesize() already has a "no chunks retrieved"
path, but that path's message implies the SOPs just don't cover the
topic, which would be actively misleading during a real DB outage. The
db_unavailable flag lets synthesize give an honest, distinct message
for this specific case.
"""
from app.graph.state import GraphState
from app.reliability.circuit_breaker import CircuitOpenError
from app.retrieval.retrievers.hybrid_retriever import hybrid_search

RETRIEVE_TOP_N = 20


def retrieve(state: GraphState) -> dict:
    """Uses the resolved query if a prior node (resolve_follow_up) set
    one, otherwise falls back to the raw user_message directly."""
    query = state.get("resolved_query") or state["user_message"]

    try:
        results = hybrid_search(query, limit=RETRIEVE_TOP_N)
    except CircuitOpenError:
        return {"retrieved_chunks": [], "db_unavailable": True}

    return {"retrieved_chunks": results}

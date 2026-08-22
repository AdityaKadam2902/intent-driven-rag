"""
Rerank node - narrows the top 20 retrieved chunks down to the top 5
that actually go to synthesis.

HONEST STATE OF THIS NODE: hybrid_search() already returns results
sorted by combined_score (vector + keyword, per its own SQL ORDER BY),
so today this is a straightforward slice to the top N - not a real
second-pass reranker model (e.g. a cross-encoder) re-scoring the
candidates with a different, often more accurate signal. A real
cross-encoder reranker is a legitimate future upgrade (see the empty
app/retrieval/rerankers/cross_encoder.py stub in the folder structure)
but was not built now to avoid adding another model/dependency before
the simpler version is proven to need it. This node exists as a
named, separate step specifically so that upgrade can slot in later
without changing the graph structure around it - only this function's
internals would change.
"""
from app.graph.state import GraphState

RERANK_TOP_N = 5


def rerank(state: GraphState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    top_chunks = chunks[:RERANK_TOP_N]
    return {"retrieved_chunks": top_chunks}

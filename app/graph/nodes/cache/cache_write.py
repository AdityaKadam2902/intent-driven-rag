"""
Cache write-through node - runs after synthesize on a cache miss.

What must NEVER be cached (each would be served confidently to the next
person long after the cause is gone):
  - answers produced while Postgres was unavailable
  - Low-confidence answers. This is the important one: it covers the
    Gemini-outage fallback ("I'm having trouble generating a written
    answer..."), the "I don't have information on this" answer, and
    weak-retrieval answers. All of them are Low by construction.
  - anything without an answer

Layer 0 only needs the document versions. Layer 1 additionally needs
the query embedding and entities; if cache_lookup couldn't compute
those (embedder/Gemini down), only Layer 0 is written.
"""
from app.cache.layer0_exact import set_layer0
from app.cache.layer1_semantic import set_layer1
from app.graph.state import GraphState


def cache_write(state: GraphState) -> dict:
    if state.get("cache_hit") or state.get("db_unavailable"):
        return {}

    answer = state.get("answer")
    if not answer or state.get("confidence") not in ("High", "Medium"):
        return {}

    document_versions = state.get("_cache_document_versions")
    if document_versions is None:
        return {}

    response = {
        "answer": answer,
        "confidence": state.get("confidence"),
        "sources": state.get("sources"),
    }
    query = state.get("resolved_query") or state["user_message"]

    set_layer0(query, document_versions, response)

    query_embedding = state.get("_cache_query_embedding")
    query_entities = state.get("_cache_query_entities")
    if query_embedding is not None and query_entities is not None:
        set_layer1(query_embedding, query_entities, document_versions, response)

    return {}

"""
Cache lookup node - sits between the intention router and retrieve.
Checks Layer 0 (exact match) then Layer 1 (entity-guarded semantic).

Hit:  cache_hit=True plus the cached answer/confidence/sources placed
      straight into state; the graph routes to END (see builder.py).
Miss: cache_hit=False, and whatever was computed along the way is
      stashed in state so cache_write can reuse it instead of paying
      for the embedding / entity extraction twice.

GOLDEN RULE: the cache is an optimization, never a dependency. Every
failure in here - Postgres, Redis, the embedder, Gemini - degrades to
"cache miss" and the normal pipeline runs. Nothing in this node may
raise into the graph.
"""
from app.cache.layer0_exact import get_layer0
from app.cache.layer1_semantic import get_layer1
from app.graph.state import GraphState
from app.ingestion.embeddings.embedder import embed_query
from app.repositories.document_versions import get_all_document_versions


def _extract_query_entities(query: str) -> dict:
    """Entity tags for the query text, using the same fixed-vocabulary
    tagger as ingestion so query-side and chunk-side tags agree.
    Imported lazily to keep module import cheap."""
    from app.ingestion.processors.entity_tagger import tag_entities

    tags = tag_entities(query)
    return {
        "people": tags.people,
        "roles": tags.roles,
        "topics": tags.fine_grained_topics,
    }


def cache_lookup(state: GraphState) -> dict:
    query = state.get("resolved_query") or state["user_message"]

    try:
        document_versions = get_all_document_versions()
    except Exception:
        # Postgres unavailable (breaker open or a raw connection error
        # before it trips). Skip caching this turn entirely; retrieve()
        # will deal with the database on its own.
        return {"cache_hit": False}

    l0 = get_layer0(query, document_versions)
    if l0 is not None:
        return {"cache_hit": True, **l0}

    # From here on, always keep the versions so cache_write can at
    # least populate Layer 0, even if the Layer 1 work below fails.
    miss: dict = {"cache_hit": False, "_cache_document_versions": document_versions}

    try:
        query_embedding = embed_query(query)
        query_entities = _extract_query_entities(query)
    except Exception:
        # Embedder or Gemini unavailable: Layer 1 is skipped, Layer 0
        # still works. (This is the path a Gemini outage takes.)
        return miss

    l1 = get_layer1(query_embedding, query_entities, document_versions)
    if l1 is not None:
        return {"cache_hit": True, **l1}

    miss["_cache_query_embedding"] = query_embedding
    miss["_cache_query_entities"] = query_entities
    return miss

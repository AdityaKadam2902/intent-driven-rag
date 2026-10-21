"""
Layer 1 cache - entity-guarded semantic match.

THE CORE DESIGN PROBLEM THIS SOLVES: a naive semantic cache stores a
question's embedding and returns the cached answer for any new
question that lands close enough in vector space. Published research
found a specific, repeatable failure: two questions that share
sentence structure but differ in one critical detail (a name, a role,
a time period) can score above a 0.95 similarity threshold and still
return the WRONG cached answer.

Concretely, for this project: "Who approves a Team Lead's leave?" and
"Who approves an employee's leave?" are structurally almost identical
but have different correct answers in the real SOP data. A pure
similarity cache is exactly the kind of system that would confidently
serve one for the other.

THE FIX: vector similarity finds CANDIDATE matches, but a candidate is
only treated as a real hit if the entity tags of the new query also
match the cached query's entity tags. This reuses the entity tagging
already produced during ingestion (people/roles/topics) - no separate
system, no extra model, just enforcing agreement on the entities
before trusting a similarity match.

Cache entries here are stored with their embedding vector and entity
tags; because Redis doesn't do vector similarity search natively, this
implementation keeps a small local index of recent entries and
computes cosine similarity in Python - reasonable at the personal
project scale this system targets (hundreds to low thousands of
cached queries, not millions).
"""
import json
import math
import time

from app.cache.redis_client import get_redis_client, with_redis_breaker
from app.config.engine_config import get_engine_config
from app.reliability.circuit_breaker import CircuitOpenError

LAYER1_INDEX_KEY = "cache:l1:index"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _entities_match(entities_a: dict, entities_b: dict) -> bool:
    """The entity guard itself: people, roles, and topics must all
    match exactly (as sets) between the candidate and the cached
    entry. Deliberately strict - a partial overlap is NOT treated as a
    match, since that's exactly the ambiguity that causes the
    false-positive failure mode this layer exists to prevent."""
    for key in ("people", "roles", "topics"):
        if set(entities_a.get(key, [])) != set(entities_b.get(key, [])):
            return False
    return True


def get_layer1(
    query_embedding: list[float],
    query_entities: dict,
    document_versions: dict[str, str],
) -> dict | None:
    """Searches the small local index of recent Layer 1 entries for a
    candidate above the similarity threshold AND with matching
    entities. Returns the cached response, or None if nothing
    qualifies (including if Redis is unavailable - fails open)."""
    config = get_engine_config()

    try:
        raw_entries = with_redis_breaker(get_redis_client().lrange, LAYER1_INDEX_KEY, 0, -1)
    except CircuitOpenError:
        return None
    except Exception:
        return None

    best_score = 0.0
    best_response = None

    for raw in raw_entries:
        entry = json.loads(raw)

        if entry.get("document_versions") != document_versions:
            continue

        score = _cosine_similarity(query_embedding, entry["embedding"])
        if score < config.cache_semantic_similarity_threshold:
            continue

        if not _entities_match(query_entities, entry["entities"]):
            continue

        if score > best_score:
            best_score = score
            best_response = entry["response"]

    return best_response


def set_layer1(
    query_embedding: list[float],
    query_entities: dict,
    document_versions: dict[str, str],
    response: dict,
    max_index_size: int = 500,
) -> None:
    """Write-through after a real answer. Keeps the local index capped
    at max_index_size (oldest entries dropped) - a simple bound so the
    per-query linear scan in get_layer1 stays fast at personal-project
    scale."""
    entry = {
        "embedding": query_embedding,
        "entities": query_entities,
        "document_versions": document_versions,
        "response": response,
        "cached_at": time.time(),
    }
    try:
        client = get_redis_client()
        with_redis_breaker(client.lpush, LAYER1_INDEX_KEY, json.dumps(entry))
        with_redis_breaker(client.ltrim, LAYER1_INDEX_KEY, 0, max_index_size - 1)
        config = get_engine_config()
        with_redis_breaker(client.expire, LAYER1_INDEX_KEY, config.cache_ttl_seconds)
    except (CircuitOpenError, Exception):
        pass

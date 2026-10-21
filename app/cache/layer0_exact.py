"""
Layer 0 cache - exact match, keyed by normalized query text plus the
document versions currently in play. This is the cheapest, zero-risk
layer: if the exact same question is asked again, and nothing has
changed, return the exact same cached answer instantly.

Why include document versions in the key, not just the query text:
this is what makes cache invalidation automatic and correct. If any
relevant document updates, its version changes, the key naturally
changes too, and the old cached entry is simply never looked up again
- no explicit invalidation logic needed for this layer specifically.
(Layer 1's entries DO need explicit invalidation since they're not
keyed by exact version - see cache/invalidation.py.)
"""
import hashlib
import json

from app.cache.redis_client import get_redis_client, with_redis_breaker
from app.config.engine_config import get_engine_config
from app.reliability.circuit_breaker import CircuitOpenError

LAYER0_KEY_PREFIX = "cache:l0"


def _normalize_query(query: str) -> str:
    """Lowercase, strip, collapse whitespace - so trivial formatting
    differences ('Who approves WFH?' vs 'who approves wfh') still hit
    the same cache entry."""
    return " ".join(query.lower().strip().split())


def build_layer0_key(query: str, document_versions: dict[str, str]) -> str:
    """document_versions: {document_title: version} for every document
    that contributed to (or could contribute to) this answer. Sorted
    before hashing so key order never matters."""
    normalized = _normalize_query(query)
    versions_str = json.dumps(dict(sorted(document_versions.items())))
    combined = f"{normalized}|{versions_str}"
    digest = hashlib.sha256(combined.encode()).hexdigest()
    return f"{LAYER0_KEY_PREFIX}:{digest}"


def get_layer0(query: str, document_versions: dict[str, str]) -> dict | None:
    """Returns the cached response dict, or None on a miss OR if Redis
    is unavailable - a Redis outage must look identical to a cache
    miss from the caller's perspective, never an error."""
    key = build_layer0_key(query, document_versions)
    try:
        raw = with_redis_breaker(get_redis_client().get, key)
    except CircuitOpenError:
        return None
    except Exception:
        return None
    if raw is None:
        return None
    return json.loads(raw)


def set_layer0(query: str, document_versions: dict[str, str], response: dict) -> None:
    """Write-through after a real answer is generated. Failures here
    are silently swallowed - a failed cache WRITE should never break
    the response the user is about to receive."""
    key = build_layer0_key(query, document_versions)
    config = get_engine_config()
    try:
        with_redis_breaker(
            get_redis_client().set, key, json.dumps(response), ex=config.cache_ttl_seconds
        )
    except (CircuitOpenError, Exception):
        pass

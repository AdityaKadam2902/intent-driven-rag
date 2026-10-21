"""
Redis connection handling for the caching layer.

Owns its OWN circuit breaker, separate from both the Gemini breaker
(app/llm/gemini.py) and the Postgres breaker (app/database/postgres.py)
- three independent dependencies, three independent failure domains.

Critically, this breaker protects the SYSTEM from a Redis outage, but
Redis failures should always be non-fatal - caching is a performance
optimization, never a correctness requirement. If Redis is down, the
system should transparently skip the cache and hit the real pipeline
every time, not degrade the user experience. See cache_lookup.py and
cache_write.py for how callers handle this (catch and continue, never
propagate a Redis failure up to the user).
"""
import redis

from app.config.engine_config import get_engine_config
from app.reliability.circuit_breaker import CircuitBreaker

_client: redis.Redis | None = None
_redis_breaker: CircuitBreaker | None = None


def _get_breaker() -> CircuitBreaker:
    global _redis_breaker
    if _redis_breaker is None:
        # Deliberately more aggressive than Gemini/Postgres breakers -
        # a low threshold and short cooldown, since caching is purely
        # an optimization. We want to stop wasting time on a failing
        # Redis quickly and get back to normal (cache-skipped)
        # operation, not linger trying.
        _redis_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=15.0)
    return _redis_breaker


def get_redis_client() -> redis.Redis:
    global _client
    if _client is None:
        config = get_engine_config()
        _client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _client


def with_redis_breaker(fn, *args, **kwargs):
    """Runs fn through the shared Redis circuit breaker. Raises
    CircuitOpenError if open - callers must catch this and gracefully
    skip caching, never let it interrupt the real pipeline."""
    return _get_breaker().call(fn, *args, **kwargs)


def get_redis_breaker_state() -> str:
    return _get_breaker().state.value


def reset_redis_breaker():
    global _redis_breaker
    _redis_breaker = None

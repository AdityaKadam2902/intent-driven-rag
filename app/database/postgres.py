"""
Postgres connection handling. Reads connection details from environment
variables (.env).

Owns a single, shared CircuitBreaker instance for Postgres calls -
separate from the Gemini breaker in app/llm/gemini.py, deliberately
(two independent failure domains, see that file's module docstring
for the full reasoning). Breaker thresholds are read from EngineConfig
rather than hardcoded - this used to be
CircuitBreaker(failure_threshold=5, recovery_timeout_seconds=20.0)
hardcoded directly here; now it's tunable via
ENGINE_POSTGRES_BREAKER_FAILURE_THRESHOLD /
ENGINE_POSTGRES_BREAKER_RECOVERY_TIMEOUT_SECONDS env vars, same
defaults as before.
"""
import os

import psycopg2
from dotenv import load_dotenv

from app.config.engine_config import get_engine_config
from app.reliability.circuit_breaker import CircuitBreaker

load_dotenv()

_postgres_breaker: CircuitBreaker | None = None


def _get_breaker() -> CircuitBreaker:
    global _postgres_breaker
    if _postgres_breaker is None:
        config = get_engine_config()
        _postgres_breaker = CircuitBreaker(
            failure_threshold=config.postgres_breaker_failure_threshold,
            recovery_timeout_seconds=config.postgres_breaker_recovery_timeout_seconds,
        )
    return _postgres_breaker


def get_connection():
    """Returns a new psycopg2 connection. NOT wrapped in the circuit
    breaker itself - the breaker wraps the actual query execution (see
    with_db_breaker below), since a connection could succeed but a
    query could still time out or fail."""
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ.get("POSTGRES_DB", "meldep_sop_dev"),
        user=os.environ.get("POSTGRES_USER", "meldep_dev"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
    )


def with_db_breaker(fn, *args, **kwargs):
    """Runs fn(*args, **kwargs) through the shared Postgres circuit
    breaker. Use this to wrap any function that does real DB work."""
    return _get_breaker().call(fn, *args, **kwargs)


def get_db_breaker_state() -> str:
    return _get_breaker().state.value


def reset_db_breaker():
    """Testing/ops escape hatch. Resets the cached breaker instance
    itself, not just its state."""
    global _postgres_breaker
    _postgres_breaker = None

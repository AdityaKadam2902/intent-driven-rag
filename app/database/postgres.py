"""
Postgres connection handling. Reads connection details from environment
variables (.env), same pattern as app/llm/gemini.py reading
GEMINI_API_KEY.

Also owns a single, shared CircuitBreaker instance for Postgres calls -
separate from the Gemini breaker in app/llm/gemini.py, deliberately.
These protect against two independent dependencies; if Postgres is
down, that says nothing about whether Gemini is up, and vice versa.
Sharing one breaker across both would mean a Postgres outage could
incorrectly trip the Gemini circuit (or the reverse) - two genuinely
separate failure domains need two separate breakers, same reasoning
as why the Gemini breaker itself is never Redis-backed.
"""
import os

import psycopg2
from dotenv import load_dotenv

from app.reliability.circuit_breaker import CircuitBreaker

load_dotenv()

# Deliberately separate from the Gemini breaker (app/llm/gemini.py) -
# see module docstring. Slightly more tolerant threshold than the
# Gemini breaker (5 vs 3) since a single slow query is more likely to
# be transient than a genuine outage, but still trips well before a
# user notices repeated hangs.
_postgres_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout_seconds=20.0)


def get_connection():
    """Returns a new psycopg2 connection. Caller is responsible for
    closing it (or use it as a context manager). NOT wrapped in the
    circuit breaker itself - the breaker wraps the actual query
    execution (see with_db_breaker below), since a connection could
    succeed but a query could still time out or fail."""
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ.get("POSTGRES_DB", "meldep_sop_dev"),
        user=os.environ.get("POSTGRES_USER", "meldep_dev"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
    )


def with_db_breaker(fn, *args, **kwargs):
    """Runs fn(*args, **kwargs) through the shared Postgres circuit
    breaker. Use this to wrap any function that does real DB work
    (connect + query + fetch), not just the connection itself - a
    connection can succeed while the actual query still fails or
    hangs, and that's the failure this needs to catch.

    Raises CircuitOpenError if the breaker is open - callers are
    responsible for catching this and returning a graceful,
    user-facing message, never letting it surface as a raw DB
    exception."""
    return _postgres_breaker.call(fn, *args, **kwargs)


def get_db_breaker_state() -> str:
    """Exposed for observability - lets the web app or logs show
    current DB breaker state."""
    return _postgres_breaker.state.value


def reset_db_breaker():
    """Testing/ops escape hatch."""
    _postgres_breaker.reset()

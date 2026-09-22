"""
Thin wrapper around the Gemini API client.

Kept separate from the fact extractor (and future entity tagger, etc.)
so the actual model/SDK is a swap-out point.

Owns a single, shared CircuitBreaker instance for all Gemini calls
across the graph. Breaker thresholds are read from EngineConfig
(app/config/engine_config.py) rather than hardcoded here - this used
to be CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=30.0)
hardcoded directly in this file; now it's tunable via
ENGINE_GEMINI_BREAKER_FAILURE_THRESHOLD / ENGINE_GEMINI_BREAKER_RECOVERY_TIMEOUT_SECONDS
env vars without touching code, with the same defaults as before.
"""
import os

from dotenv import load_dotenv
from google import genai

from app.config.engine_config import get_engine_config
from app.reliability.circuit_breaker import CircuitBreaker, CircuitOpenError

load_dotenv()

_client: genai.Client | None = None

# NOTE: gemini-2.5-flash was shut down early (June 2026, ahead of its
# originally announced date) - treat this string as something to
# periodically re-check, not a one-time setting.
DEFAULT_MODEL = "gemini-3.1-flash-lite"

# Built lazily from EngineConfig on first use, not at import time - so
# a test that overrides ENGINE_* env vars before first use gets the
# right values, rather than whatever was cached at module import.
_gemini_breaker: CircuitBreaker | None = None


def _get_breaker() -> CircuitBreaker:
    global _gemini_breaker
    if _gemini_breaker is None:
        config = get_engine_config()
        _gemini_breaker = CircuitBreaker(
            failure_threshold=config.gemini_breaker_failure_threshold,
            recovery_timeout_seconds=config.gemini_breaker_recovery_timeout_seconds,
        )
    return _gemini_breaker


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found in environment. "
                "Create a .env file in the project root with: "
                "GEMINI_API_KEY=your_key_here"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def generate_content_safe(**kwargs):
    """Wraps client.models.generate_content() through the shared
    circuit breaker. Every node that calls Gemini should use this
    instead of calling client.models.generate_content() directly."""
    client = get_client()
    return _get_breaker().call(client.models.generate_content, **kwargs)


def get_breaker_state() -> str:
    return _get_breaker().state.value


def reset_breaker():
    """Testing/ops escape hatch. Also resets the cached breaker instance
    itself, not just its state - so a test changing ENGINE_* env vars
    between runs actually picks up the new thresholds."""
    global _gemini_breaker
    _gemini_breaker = None

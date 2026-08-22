"""
Thin wrapper around the Gemini API client.

Kept separate from the fact extractor (and future entity tagger, etc.)
so the actual model/SDK is a swap-out point - if the model name or SDK
changes later, this is the one file that needs to change.

Reads GEMINI_API_KEY from the environment (.env file, loaded via
python-dotenv). Never hardcode the key here or anywhere else.

Also owns a single, shared CircuitBreaker instance for all Gemini
calls across the graph (classify_intention, resolve_follow_up,
synthesize all route through generate_content_safe below). One shared
breaker, not one per node, because they're all protecting against the
same underlying dependency (the Gemini API) - if it's failing for one
node, it's failing for all of them, and they should trip together.
"""
import os

from dotenv import load_dotenv
from google import genai

from app.reliability.circuit_breaker import CircuitBreaker, CircuitOpenError

load_dotenv()

_client: genai.Client | None = None

# Model choice lives here, in one place - update this string if the
# model version needs to change, nothing else in the codebase should
# need to know the model name.
# NOTE: gemini-2.5-flash was shut down early (June 2026, ahead of its
# originally announced date) - Google has done this before, so treat
# this string as something to periodically re-check, not a one-time
# setting. gemini-3.1-flash-lite is GA and stable (no shutdown date
# announced as of July 2026), and well-suited to structured extraction
# specifically - it's the cost-efficient tier and this is a precision
# task, not one needing Pro-level reasoning.
DEFAULT_MODEL = "gemini-3.1-flash-lite"

# Shared across every node that calls Gemini - deliberately in-process,
# never Redis-backed. See app/reliability/circuit_breaker.py's
# docstring for why this matters.
_gemini_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=30.0)


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
    instead of calling client.models.generate_content() directly, so
    the breaker actually protects every real call path, not just some
    of them.

    Raises CircuitOpenError if the breaker is open - callers (the
    graph nodes) are responsible for catching this and returning a
    graceful fallback state update, never letting it surface as a raw
    exception/500 error to the user. See
    app/graph/nodes/generation/synthesize.py for the fallback pattern.
    """
    client = get_client()
    return _gemini_breaker.call(client.models.generate_content, **kwargs)


def get_breaker_state() -> str:
    """Exposed for observability/debugging - lets the web app or logs
    show current breaker state without reaching into the private
    module-level instance directly."""
    return _gemini_breaker.state.value


def reset_breaker():
    """Testing/ops escape hatch - not used in normal operation."""
    _gemini_breaker.reset()

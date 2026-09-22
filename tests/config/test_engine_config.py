"""
Proves the engine config actually controls the breakers, not just that
both exist independently. Tests both: (1) defaults match what was
previously hardcoded, so introducing the config layer changed nothing
about runtime behavior, and (2) env var overrides genuinely take effect.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.config.engine_config import get_engine_config


def setup_function():
    get_engine_config.cache_clear()
    import app.llm.gemini as gemini_module
    import app.database.postgres as postgres_module
    gemini_module._gemini_breaker = None
    postgres_module._postgres_breaker = None
    for key in [
        "ENGINE_GEMINI_BREAKER_FAILURE_THRESHOLD",
        "ENGINE_GEMINI_BREAKER_RECOVERY_TIMEOUT_SECONDS",
        "ENGINE_POSTGRES_BREAKER_FAILURE_THRESHOLD",
        "ENGINE_RETRIEVE_TOP_N",
        "ENGINE_RERANK_TOP_N",
    ]:
        os.environ.pop(key, None)


def teardown_function():
    setup_function()


def test_defaults_match_previously_hardcoded_values():
    """Introducing the config layer must not silently change behavior -
    every default here must match what was hardcoded before this
    config existed."""
    config = get_engine_config()

    assert config.retrieve_top_n == 20
    assert config.rerank_top_n == 5
    assert config.high_confidence_threshold == 0.6
    assert config.medium_confidence_threshold == 0.4
    assert config.gemini_breaker_failure_threshold == 3
    assert config.gemini_breaker_recovery_timeout_seconds == 30.0
    assert config.postgres_breaker_failure_threshold == 5
    assert config.postgres_breaker_recovery_timeout_seconds == 20.0


def test_vector_and_text_weights_sum_to_one():
    config = get_engine_config()
    config.validate_weights()


def test_gemini_breaker_uses_config_threshold():
    """The actual integration proof: does app/llm/gemini.py's breaker
    genuinely read from config, not just have a config object sitting
    unused nearby?"""
    os.environ["ENGINE_GEMINI_BREAKER_FAILURE_THRESHOLD"] = "1"
    get_engine_config.cache_clear()

    import app.llm.gemini as gemini_module
    gemini_module._gemini_breaker = None

    breaker = gemini_module._get_breaker()

    assert breaker.failure_threshold == 1


def test_postgres_breaker_uses_config_threshold():
    os.environ["ENGINE_POSTGRES_BREAKER_FAILURE_THRESHOLD"] = "2"
    get_engine_config.cache_clear()

    import app.database.postgres as postgres_module
    postgres_module._postgres_breaker = None

    breaker = postgres_module._get_breaker()

    assert breaker.failure_threshold == 2


def test_gemini_and_postgres_breakers_can_have_different_thresholds_via_config():
    """Confirms the two breakers stay independently configurable, not
    accidentally merged into one shared setting."""
    os.environ["ENGINE_GEMINI_BREAKER_FAILURE_THRESHOLD"] = "7"
    os.environ["ENGINE_POSTGRES_BREAKER_FAILURE_THRESHOLD"] = "9"
    get_engine_config.cache_clear()

    import app.llm.gemini as gemini_module
    import app.database.postgres as postgres_module
    gemini_module._gemini_breaker = None
    postgres_module._postgres_breaker = None

    gemini_breaker = gemini_module._get_breaker()
    postgres_breaker = postgres_module._get_breaker()

    assert gemini_breaker.failure_threshold == 7
    assert postgres_breaker.failure_threshold == 9

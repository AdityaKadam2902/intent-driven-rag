"""
End-to-end cache behaviour through the REAL compiled graph: real
cache_lookup / cache_write nodes, real Layer 0 + Layer 1, fakeredis as
the store. Only the external systems (Gemini, Postgres, the embedder)
are patched.
"""
import os, sys
from unittest.mock import MagicMock, patch

import fakeredis
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.cache.redis_client import reset_redis_breaker
from app.graph.builder import build_graph
from app.graph.nodes.routing.classify_intention import IntentionClassification
from app.ingestion.processors.entity_tagger import EntityTags
from app.reliability.circuit_breaker import CircuitOpenError
from app.retrieval.retrievers.hybrid_retriever import RetrievedChunk

pytestmark = pytest.mark.real_cache

QUESTION = {"user_message": "who approves WFH requests"}


def _chunk(score=0.7):
    return RetrievedChunk(
        chunk_id="1", document_id="1", document_title="WFH Policy", document_version="1.0",
        heading_path="Approval", content="Team Lead and Vikas Sir approve WFH.",
        category="HR", sub_category="WFH", people=["Vikas Sir"], roles=["Team Lead"],
        fine_grained_topics=["approval"], vector_score=score, text_score=score, combined_score=score,
    )


class Env:
    """Wires every patch once per test and exposes call counters."""

    def __init__(self):
        self.redis = fakeredis.FakeRedis(decode_responses=True)
        self.versions = {"WFH Policy": "1.0"}
        self.chunks = [_chunk()]
        self.embedding = [0.1] * 8
        self.tags = EntityTags(people=["Vikas Sir"], roles=["Team Lead"], fine_grained_topics=["approval"])
        self.synth = MagicMock()
        self.synth.return_value = MagicMock(text="Team Lead and Vikas Sir approve.")
        self.search = MagicMock(side_effect=lambda *a, **k: self.chunks)

    def ask(self, state=None):
        return build_graph().invoke(dict(state or QUESTION))


@pytest.fixture
def env():
    reset_redis_breaker()
    e = Env()
    classify = MagicMock()
    classify.return_value = MagicMock(parsed=IntentionClassification(intention="informational"))
    patches = [
        patch("app.graph.nodes.routing.classify_intention.generate_content_safe", classify),
        patch("app.graph.nodes.retrieval.retrieve.hybrid_search", e.search),
        patch("app.graph.nodes.generation.synthesize.generate_content_safe", e.synth),
        patch("app.graph.nodes.cache.cache_lookup.get_all_document_versions", lambda: e.versions),
        patch("app.graph.nodes.cache.cache_lookup.embed_query", lambda q: list(e.embedding)),
        patch("app.ingestion.processors.entity_tagger.tag_entities", lambda q, **k: e.tags),
        patch("app.cache.layer0_exact.get_redis_client", lambda: e.redis),
        patch("app.cache.layer1_semantic.get_redis_client", lambda: e.redis),
    ]
    for p in patches:
        p.start()
    yield e
    for p in patches:
        p.stop()
    reset_redis_breaker()


def test_second_identical_question_is_served_from_cache(env):
    first = env.ask()
    second = env.ask()

    assert first["answer"] == second["answer"] == "Team Lead and Vikas Sir approve."
    assert not first.get("cache_hit")
    assert second["cache_hit"] is True
    assert env.synth.call_count == 1     # no second LLM call
    assert env.search.call_count == 1    # no second retrieval
    assert second["sources"] == first["sources"]


def test_document_version_change_invalidates_the_cache(env):
    env.ask()
    env.versions = {"WFH Policy": "2.0"}
    env.ask()

    assert env.synth.call_count == 2


def test_low_confidence_answer_is_not_cached(env):
    env.chunks = [_chunk(score=0.1)]
    env.ask()
    env.ask()

    assert env.synth.call_count == 2


def test_gemini_outage_fallback_is_not_cached_and_recovers(env):
    """The regression for the 'cached an outage message for 24h' bug."""
    env.synth.side_effect = CircuitOpenError("gemini breaker open")
    down1 = env.ask()
    down2 = env.ask()
    assert "trouble generating" in down1["answer"]
    assert down2.get("cache_hit") is not True
    assert env.synth.call_count == 2

    env.synth.side_effect = None
    env.synth.return_value = MagicMock(text="Real answer after recovery.")
    recovered = env.ask()
    assert recovered["answer"] == "Real answer after recovery."


def test_similar_question_with_same_entities_hits_layer1(env):
    env.ask()                                   # caches under embedding [0.1]*8
    env.embedding = [0.1001] * 8                # near-identical, different text hash
    second = env.ask({"user_message": "which people sign off on wfh"})

    assert second["cache_hit"] is True
    assert env.synth.call_count == 1


def test_similar_question_with_different_entities_misses(env):
    """The false-positive the entity guard exists for."""
    env.ask()
    env.embedding = [0.1001] * 8
    env.tags = EntityTags(people=[], roles=["Intern"], fine_grained_topics=["approval"])
    second = env.ask({"user_message": "who approves an intern's wfh"})

    assert not second.get("cache_hit")
    assert env.synth.call_count == 2


def test_redis_down_pipeline_still_answers(env):
    def boom(*a, **k):
        raise ConnectionError("redis down")

    for name in ("get", "set", "lrange", "lpush", "ltrim", "expire"):
        setattr(env.redis, name, boom)

    result = env.ask()

    assert result["answer"] == "Team Lead and Vikas Sir approve."
    assert not result.get("cache_hit")

import os, sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.graph.nodes.cache.cache_lookup import cache_lookup

BASE = "app.graph.nodes.cache.cache_lookup"
VERSIONS = {"WFH Policy": "1.0"}
CACHED = {"answer": "cached answer", "confidence": "High", "sources": [{"document_title": "WFH Policy"}]}
ENTITIES = {"people": [], "roles": ["Intern"], "topics": ["eligibility"]}


def _p(name, **kw):
    return patch(f"{BASE}.{name}", **kw)


def test_layer0_hit_returns_cached_answer_and_skips_everything_else():
    with _p("get_all_document_versions", return_value=VERSIONS), \
         _p("get_layer0", return_value=CACHED), \
         _p("embed_query") as emb, _p("_extract_query_entities") as ent, _p("get_layer1") as l1:
        result = cache_lookup({"user_message": "who approves wfh"})

    assert result["cache_hit"] is True
    assert result["answer"] == "cached answer"
    assert result["confidence"] == "High"
    emb.assert_not_called()      # Layer 0 hit must not pay for an embedding
    ent.assert_not_called()
    l1.assert_not_called()


def test_layer1_hit_returns_cached_answer():
    with _p("get_all_document_versions", return_value=VERSIONS), \
         _p("get_layer0", return_value=None), \
         _p("embed_query", return_value=[0.1] * 4), \
         _p("_extract_query_entities", return_value=ENTITIES), \
         _p("get_layer1", return_value=CACHED):
        result = cache_lookup({"user_message": "are interns allowed wfh"})

    assert result["cache_hit"] is True
    assert result["answer"] == "cached answer"


def test_full_miss_stashes_everything_cache_write_needs():
    with _p("get_all_document_versions", return_value=VERSIONS), \
         _p("get_layer0", return_value=None), \
         _p("embed_query", return_value=[0.1] * 4), \
         _p("_extract_query_entities", return_value=ENTITIES), \
         _p("get_layer1", return_value=None):
        result = cache_lookup({"user_message": "anything"})

    assert result["cache_hit"] is False
    assert result["_cache_document_versions"] == VERSIONS
    assert result["_cache_query_embedding"] == [0.1] * 4
    assert result["_cache_query_entities"] == ENTITIES


def test_resolved_query_is_used_instead_of_raw_follow_up():
    with _p("get_all_document_versions", return_value=VERSIONS), \
         _p("get_layer0", return_value=None) as l0, \
         _p("embed_query", return_value=[0.1]), \
         _p("_extract_query_entities", return_value=ENTITIES), \
         _p("get_layer1", return_value=None):
        cache_lookup({"user_message": "what about interns", "resolved_query": "WFH policy for interns?"})

    assert l0.call_args[0][0] == "WFH policy for interns?"


def test_postgres_failure_skips_caching_without_raising():
    with _p("get_all_document_versions", side_effect=ConnectionError("db down")), \
         _p("get_layer0") as l0:
        result = cache_lookup({"user_message": "x"})

    assert result == {"cache_hit": False}
    l0.assert_not_called()


def test_gemini_or_embedder_failure_degrades_to_layer0_only():
    """A Gemini outage must not crash the node - it's just a miss, and
    the versions are still stashed so Layer 0 can be populated."""
    with _p("get_all_document_versions", return_value=VERSIONS), \
         _p("get_layer0", return_value=None), \
         _p("embed_query", return_value=[0.1]), \
         _p("_extract_query_entities", side_effect=RuntimeError("gemini down")), \
         _p("get_layer1") as l1:
        result = cache_lookup({"user_message": "x"})

    assert result["cache_hit"] is False
    assert result["_cache_document_versions"] == VERSIONS
    assert "_cache_query_embedding" not in result
    l1.assert_not_called()

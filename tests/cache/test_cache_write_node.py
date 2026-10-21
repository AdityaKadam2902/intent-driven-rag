import os, sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.graph.nodes.cache.cache_write import cache_write

BASE = "app.graph.nodes.cache.cache_write"
VERSIONS = {"WFH Policy": "1.0"}


def _state(**over):
    s = {
        "user_message": "who approves wfh",
        "answer": "Team Lead and Vikas Sir approve.",
        "confidence": "High",
        "sources": [],
        "_cache_document_versions": VERSIONS,
        "_cache_query_embedding": [0.1],
        "_cache_query_entities": {"people": [], "roles": [], "topics": []},
    }
    s.update(over)
    return s


def _run(state):
    with patch(f"{BASE}.set_layer0") as l0, patch(f"{BASE}.set_layer1") as l1:
        result = cache_write(state)
    return result, l0, l1


def test_high_confidence_miss_writes_both_layers():
    result, l0, l1 = _run(_state())
    assert result == {}
    l0.assert_called_once()
    l1.assert_called_once()


def test_medium_confidence_is_cached_too():
    _, l0, _ = _run(_state(confidence="Medium"))
    l0.assert_called_once()


def test_low_confidence_is_never_cached():
    """Covers the Gemini-outage fallback, 'no information' answers and
    weak retrieval - all Low by construction."""
    _, l0, l1 = _run(_state(confidence="Low"))
    l0.assert_not_called()
    l1.assert_not_called()


def test_gemini_outage_fallback_answer_is_not_cached():
    fallback = _state(
        answer="I'm having trouble generating a written answer right now, but here are the SOP sections...",
        confidence="Low",
    )
    _, l0, l1 = _run(fallback)
    l0.assert_not_called()
    l1.assert_not_called()


def test_db_unavailable_is_never_cached():
    _, l0, l1 = _run(_state(db_unavailable=True))
    l0.assert_not_called()
    l1.assert_not_called()


def test_cache_hit_is_not_rewritten():
    _, l0, l1 = _run(_state(cache_hit=True))
    l0.assert_not_called()
    l1.assert_not_called()


def test_missing_versions_writes_nothing():
    s = _state()
    del s["_cache_document_versions"]
    _, l0, l1 = _run(s)
    l0.assert_not_called()
    l1.assert_not_called()


def test_missing_embedding_writes_layer0_only():
    s = _state()
    del s["_cache_query_embedding"]
    _, l0, l1 = _run(s)
    l0.assert_called_once()
    l1.assert_not_called()


def test_empty_answer_is_not_cached():
    _, l0, _ = _run(_state(answer=""))
    l0.assert_not_called()

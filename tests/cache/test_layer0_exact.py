"""
Uses fakeredis (an in-memory Redis-compatible server) instead of a
mock, so these tests exercise real Redis command behavior (TTL,
serialization round-trips) without needing a live Redis server.
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import fakeredis

from app.cache.layer0_exact import build_layer0_key, get_layer0, set_layer0


def _fake_client():
    return fakeredis.FakeRedis(decode_responses=True)


def test_exact_same_query_and_versions_is_a_hit():
    client = _fake_client()
    with patch("app.cache.layer0_exact.get_redis_client", return_value=client):
        versions = {"WFH Policy": "1.0"}
        response = {"answer": "Team Lead and Vikas Sir approve.", "confidence": "High"}

        set_layer0("who approves WFH", versions, response)
        result = get_layer0("who approves WFH", versions)

        assert result == response


def test_case_and_whitespace_differences_still_hit():
    client = _fake_client()
    with patch("app.cache.layer0_exact.get_redis_client", return_value=client):
        versions = {"WFH Policy": "1.0"}
        response = {"answer": "answer"}

        set_layer0("Who Approves WFH?", versions, response)
        result = get_layer0("  who   approves wfh?  ", versions)

        assert result == response


def test_different_query_is_a_miss():
    client = _fake_client()
    with patch("app.cache.layer0_exact.get_redis_client", return_value=client):
        versions = {"WFH Policy": "1.0"}
        set_layer0("who approves WFH", versions, {"answer": "a"})

        result = get_layer0("what is the WFH policy", versions)

        assert result is None


def test_version_change_causes_a_miss_even_for_the_same_query():
    client = _fake_client()
    with patch("app.cache.layer0_exact.get_redis_client", return_value=client):
        set_layer0("who approves WFH", {"WFH Policy": "1.0"}, {"answer": "old"})

        result = get_layer0("who approves WFH", {"WFH Policy": "2.0"})

        assert result is None


def test_key_is_order_independent_for_multiple_documents():
    key1 = build_layer0_key("test", {"A": "1", "B": "2"})
    key2 = build_layer0_key("test", {"B": "2", "A": "1"})

    assert key1 == key2


def test_redis_unavailable_returns_none_not_an_error():
    def broken_get(*args, **kwargs):
        raise ConnectionError("redis down")

    fake_client = _fake_client()
    fake_client.get = broken_get

    with patch("app.cache.layer0_exact.get_redis_client", return_value=fake_client):
        result = get_layer0("test query", {"Doc": "1.0"})

        assert result is None


def test_redis_unavailable_on_write_does_not_raise():
    def broken_set(*args, **kwargs):
        raise ConnectionError("redis down")

    fake_client = _fake_client()
    fake_client.set = broken_set

    with patch("app.cache.layer0_exact.get_redis_client", return_value=fake_client):
        set_layer0("test", {"Doc": "1.0"}, {"answer": "a"})

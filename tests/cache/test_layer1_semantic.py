"""
The critical tests here are the entity-guard ones - proving the exact
scenario this whole design exists to prevent: two similar-sounding
questions with different correct answers must NOT be confused, even
when their embeddings are nearly identical.
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import fakeredis

from app.cache.layer1_semantic import get_layer1, set_layer1


def _fake_client():
    return fakeredis.FakeRedis(decode_responses=True)


def test_high_similarity_with_matching_entities_is_a_hit():
    client = _fake_client()
    with patch("app.cache.layer1_semantic.get_redis_client", return_value=client):
        embedding = [0.1] * 768
        entities = {"people": ["Vikas Sir"], "roles": ["Team Lead"], "topics": ["approval"]}
        versions = {"WFH Policy": "1.0"}
        response = {"answer": "Team Lead and Vikas Sir approve.", "confidence": "High"}

        set_layer1(embedding, entities, versions, response)

        similar_embedding = [0.1001] * 768
        result = get_layer1(similar_embedding, entities, versions)

        assert result == response


def test_high_similarity_but_different_entities_is_a_miss():
    """THE core regression test: 'Team Lead's leave' vs 'employee's
    leave' - structurally near-identical, high cosine similarity, but
    genuinely different questions with different correct answers.
    Must be a miss, not a wrongly-confident hit."""
    client = _fake_client()
    with patch("app.cache.layer1_semantic.get_redis_client", return_value=client):
        embedding = [0.1] * 768
        versions = {"Leave Policy": "1.0"}

        team_lead_entities = {"people": [], "roles": ["Team Lead"], "topics": ["leave"]}
        set_layer1(
            embedding, team_lead_entities, versions,
            {"answer": "Vikas Sir approves Team Lead leave.", "confidence": "High"},
        )

        employee_entities = {"people": [], "roles": ["Employee"], "topics": ["leave"]}
        similar_embedding = [0.1001] * 768

        result = get_layer1(similar_embedding, employee_entities, versions)

        assert result is None


def test_low_similarity_is_a_miss_even_with_matching_entities():
    client = _fake_client()
    with patch("app.cache.layer1_semantic.get_redis_client", return_value=client):
        embedding = [0.1] * 768
        entities = {"people": [], "roles": ["Team Lead"], "topics": ["leave"]}
        versions = {"Leave Policy": "1.0"}
        set_layer1(embedding, entities, versions, {"answer": "a"})

        unrelated_embedding = [-0.1] * 768
        result = get_layer1(unrelated_embedding, entities, versions)

        assert result is None


def test_document_version_mismatch_is_a_miss():
    client = _fake_client()
    with patch("app.cache.layer1_semantic.get_redis_client", return_value=client):
        embedding = [0.1] * 768
        entities = {"people": [], "roles": ["Team Lead"], "topics": ["leave"]}
        set_layer1(embedding, entities, {"Leave Policy": "1.0"}, {"answer": "old"})

        result = get_layer1(embedding, entities, {"Leave Policy": "2.0"})

        assert result is None


def test_partial_entity_overlap_is_still_a_miss():
    """Entity matching is deliberately strict (exact set match), not
    'any overlap counts' - a partial match is exactly the kind of
    ambiguity this layer exists to reject."""
    client = _fake_client()
    with patch("app.cache.layer1_semantic.get_redis_client", return_value=client):
        embedding = [0.1] * 768
        versions = {"WFH Policy": "1.0"}
        cached_entities = {"people": ["Vikas Sir"], "roles": ["Team Lead"], "topics": ["approval"]}
        set_layer1(embedding, cached_entities, versions, {"answer": "a"})

        query_entities = {"people": [], "roles": ["Team Lead"], "topics": ["approval"]}
        result = get_layer1(embedding, query_entities, versions)

        assert result is None


def test_redis_unavailable_returns_none_not_an_error():
    def broken_lrange(*args, **kwargs):
        raise ConnectionError("redis down")

    fake_client = _fake_client()
    fake_client.lrange = broken_lrange

    with patch("app.cache.layer1_semantic.get_redis_client", return_value=fake_client):
        result = get_layer1([0.1] * 768, {"roles": ["Team Lead"]}, {"Doc": "1.0"})

        assert result is None


def test_redis_unavailable_on_write_does_not_raise():
    def broken_lpush(*args, **kwargs):
        raise ConnectionError("redis down")

    fake_client = _fake_client()
    fake_client.lpush = broken_lpush

    with patch("app.cache.layer1_semantic.get_redis_client", return_value=fake_client):
        set_layer1([0.1] * 768, {"roles": ["Team Lead"]}, {"Doc": "1.0"}, {"answer": "a"})

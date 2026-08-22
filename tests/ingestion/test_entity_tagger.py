"""
Mocked tests - check that our code calls Gemini correctly and handles
the response correctly, without needing a real API key. See
scripts/test_entity_tagging_live.py for the real, live version.
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.ingestion.processors.entity_tagger import (
    FINE_GRAINED_TOPICS,
    EntityTaggingResult,
    tag_entities,
    tag_entities_for_chunks,
)


def _fake_response(result: EntityTaggingResult):
    fake = MagicMock()
    fake.parsed = result
    return fake


@patch("app.ingestion.processors.entity_tagger.get_client")
def test_tag_entities_returns_people_and_roles(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response(
        EntityTaggingResult(
            people=["Vikas Sir", "Sandhya Mam"],
            roles=["Team Lead"],
            fine_grained_topics=["approval"],
        )
    )
    mock_get_client.return_value = mock_client

    tags = tag_entities("some leave content", doc_title="Leave Policy")

    assert "Vikas Sir" in tags.people
    assert "Sandhya Mam" in tags.people
    assert "Team Lead" in tags.roles
    assert "approval" in tags.fine_grained_topics


@patch("app.ingestion.processors.entity_tagger.get_client")
def test_category_and_sub_category_come_from_metadata_not_ai(mock_get_client):
    """category/sub_category must be passed through from the document's
    own metadata, not invented by the model - the mock response doesn't
    even include them, proving they come from the function arguments."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response(
        EntityTaggingResult(people=[], roles=[], fine_grained_topics=[])
    )
    mock_get_client.return_value = mock_client

    tags = tag_entities(
        "content", doc_title="WFH Policy", category="HR", sub_category="WFH"
    )

    assert tags.category == "HR"
    assert tags.sub_category == "WFH"


@patch("app.ingestion.processors.entity_tagger.get_client")
def test_casing_is_normalized(mock_get_client):
    """Regression test for the exact bug found in live testing:
    'intern' vs 'Intern' must become the same normalized value, or
    cross-document search silently misses matches."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response(
        EntityTaggingResult(people=[], roles=["intern", "INTERN", "Intern"], fine_grained_topics=[])
    )
    mock_get_client.return_value = mock_client

    tags = tag_entities("content", doc_title="X")

    # all three variants must normalize to the exact same string
    assert len(set(tags.roles)) == 1
    assert tags.roles[0] == "Intern"


@patch("app.ingestion.processors.entity_tagger.get_client")
def test_topics_outside_fixed_list_are_rejected(mock_get_client):
    """Regression test for the drift problem found in live testing:
    even if the model doesn't perfectly follow the constraint and
    returns something outside FINE_GRAINED_TOPICS, our code must filter
    it out rather than trust the model blindly."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response(
        EntityTaggingResult(
            people=[],
            roles=[],
            fine_grained_topics=["approval", "some_made_up_topic_the_model_invented"],
        )
    )
    mock_get_client.return_value = mock_client

    tags = tag_entities("content", doc_title="X")

    assert "approval" in tags.fine_grained_topics
    assert "some_made_up_topic_the_model_invented" not in tags.fine_grained_topics


@patch("app.ingestion.processors.entity_tagger.get_client")
def test_prompt_includes_the_full_fixed_topic_list(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response(
        EntityTaggingResult(people=[], roles=[], fine_grained_topics=[])
    )
    mock_get_client.return_value = mock_client

    tag_entities("content", doc_title="X")

    args, kwargs = mock_client.models.generate_content.call_args
    prompt_sent = kwargs["contents"]
    for topic in FINE_GRAINED_TOPICS:
        assert topic in prompt_sent


@patch("app.ingestion.processors.entity_tagger.get_client")
def test_tag_entities_uses_deterministic_temperature(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response(
        EntityTaggingResult(people=[], roles=[], fine_grained_topics=[])
    )
    mock_get_client.return_value = mock_client

    tag_entities("content", doc_title="X")

    _, kwargs = mock_client.models.generate_content.call_args
    assert kwargs["config"].temperature == 0.0


@patch("app.ingestion.processors.entity_tagger.get_client")
@patch("app.ingestion.processors.entity_tagger.time.sleep")
def test_tag_entities_for_chunks_paces_calls_but_not_before_the_first(
    mock_sleep, mock_get_client
):
    """Regression test for the rate-limit lesson learned on the fact
    extractor: must sleep BETWEEN calls, but not waste time sleeping
    before the very first one."""
    from app.ingestion.processors.chunker import Chunk

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response(
        EntityTaggingResult(people=[], roles=[], fine_grained_topics=[])
    )
    mock_get_client.return_value = mock_client

    chunks = [
        Chunk(heading_path="A", content="a", chunk_index=0),
        Chunk(heading_path="B", content="b", chunk_index=1),
        Chunk(heading_path="C", content="c", chunk_index=2),
    ]

    tag_entities_for_chunks(chunks, doc_title="X", category="HR", sub_category="WFH")

    assert mock_sleep.call_count == 2  # 3 chunks, 2 gaps between them


@patch("app.ingestion.processors.entity_tagger.get_client")
def test_tag_entities_for_chunks_passes_category_to_every_chunk(mock_get_client):
    """category/sub_category are document-level, not per-chunk - every
    chunk in a document should get the same values."""
    from app.ingestion.processors.chunker import Chunk

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response(
        EntityTaggingResult(people=[], roles=[], fine_grained_topics=[])
    )
    mock_get_client.return_value = mock_client

    chunks = [
        Chunk(heading_path="A", content="a", chunk_index=0),
        Chunk(heading_path="B", content="b", chunk_index=1),
    ]

    results = tag_entities_for_chunks(
        chunks, doc_title="X", category="Engineer", sub_category="SDLC"
    )

    for tags in results.values():
        assert tags.category == "Engineer"
        assert tags.sub_category == "SDLC"

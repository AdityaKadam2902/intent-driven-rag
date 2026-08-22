import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.graph.nodes.generation.synthesize import (
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    synthesize,
)
from app.retrieval.retrievers.hybrid_retriever import RetrievedChunk


def _fake_chunk(combined_score, title="WFH Policy", heading="Approval"):
    return RetrievedChunk(
        chunk_id="1", document_id="1", document_title=title, document_version="1.0",
        heading_path=heading, content="WFH requests need approval from the Team Lead and Vikas Sir.",
        category="HR", sub_category="WFH", people=["Vikas Sir"], roles=["Team Lead"],
        fine_grained_topics=["approval"],
        vector_score=combined_score, text_score=combined_score, combined_score=combined_score,
    )


def _fake_llm_response(text: str):
    fake = MagicMock()
    fake.text = text
    return fake


def test_no_chunks_returns_low_confidence_no_llm_call():
    """When nothing was retrieved, must not call the LLM at all - just
    return a clear 'I don't know' style answer directly."""
    with patch("app.graph.nodes.generation.synthesize.get_client") as mock_get_client:
        result = synthesize({"user_message": "test", "retrieved_chunks": []})

        assert result["confidence"] == "Low"
        assert result["sources"] == []
        mock_get_client.assert_not_called()


@patch("app.graph.nodes.generation.synthesize.get_client")
def test_high_confidence_when_top_score_above_threshold(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_llm_response("Vikas Sir and Team Lead approve WFH.")
    mock_get_client.return_value = mock_client

    chunks = [_fake_chunk(HIGH_CONFIDENCE_THRESHOLD + 0.01)]
    result = synthesize({"user_message": "who approves WFH", "retrieved_chunks": chunks})

    assert result["confidence"] == "High"


@patch("app.graph.nodes.generation.synthesize.get_client")
def test_medium_confidence_in_middle_range(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_llm_response("answer")
    mock_get_client.return_value = mock_client

    chunks = [_fake_chunk((HIGH_CONFIDENCE_THRESHOLD + MEDIUM_CONFIDENCE_THRESHOLD) / 2)]
    result = synthesize({"user_message": "test", "retrieved_chunks": chunks})

    assert result["confidence"] == "Medium"


@patch("app.graph.nodes.generation.synthesize.get_client")
def test_low_confidence_below_threshold(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_llm_response("answer")
    mock_get_client.return_value = mock_client

    chunks = [_fake_chunk(MEDIUM_CONFIDENCE_THRESHOLD - 0.05)]
    result = synthesize({"user_message": "test", "retrieved_chunks": chunks})

    assert result["confidence"] == "Low"


@patch("app.graph.nodes.generation.synthesize.get_client")
def test_sources_include_document_and_heading(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_llm_response("answer")
    mock_get_client.return_value = mock_client

    chunks = [_fake_chunk(0.7, title="WFH Policy", heading="General WFH Policy - FAQ")]
    result = synthesize({"user_message": "test", "retrieved_chunks": chunks})

    assert result["sources"][0]["document_title"] == "WFH Policy"
    assert result["sources"][0]["heading_path"] == "General WFH Policy - FAQ"


@patch("app.graph.nodes.generation.synthesize.get_client")
def test_uses_resolved_query_over_raw_message_when_present(mock_get_client):
    """For a follow-up, the prompt should ask about the RESOLVED
    question, not the ambiguous raw follow-up text."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_llm_response("answer")
    mock_get_client.return_value = mock_client

    chunks = [_fake_chunk(0.7)]
    synthesize({
        "user_message": "what about for interns",
        "resolved_query": "What is the WFH policy for interns?",
        "retrieved_chunks": chunks,
    })

    args, kwargs = mock_client.models.generate_content.call_args
    assert "What is the WFH policy for interns?" in kwargs["contents"]


@patch("app.graph.nodes.generation.synthesize.get_client")
def test_answer_comes_from_llm_response_text(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_llm_response(
        "Both the Team Lead and Vikas Sir must approve."
    )
    mock_get_client.return_value = mock_client

    chunks = [_fake_chunk(0.7)]
    result = synthesize({"user_message": "test", "retrieved_chunks": chunks})

    assert result["answer"] == "Both the Team Lead and Vikas Sir must approve."


@patch("app.graph.nodes.generation.synthesize.get_client")
def test_duplicate_sources_are_deduplicated(mock_get_client):
    """Regression test for a real issue found in live testing: several
    chunks from the top-N can point at the SAME heading/section (with
    a small document set, near-identical-scoring chunks commonly make
    the cut), and showing the same source 4 times looks like 4 pieces
    of corroborating evidence when it's really 1 - misleading, even
    though not factually wrong."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_llm_response("answer")
    mock_get_client.return_value = mock_client

    # 4 chunks, but only 2 genuinely distinct (document, version, heading) combos
    chunks = [
        _fake_chunk(0.7, title="WFH Policy", heading="General WFH Policy - FAQ"),
        _fake_chunk(0.65, title="WFH Policy", heading="General WFH Policy - FAQ"),
        _fake_chunk(0.6, title="WFH Policy", heading="General WFH Policy - FAQ"),
        _fake_chunk(0.55, title="WFH Policy", heading="Intern WFH Policy - FAQ"),
    ]
    result = synthesize({"user_message": "test", "retrieved_chunks": chunks})

    assert len(result["sources"]) == 2
    headings = {s["heading_path"] for s in result["sources"]}
    assert headings == {"General WFH Policy - FAQ", "Intern WFH Policy - FAQ"}

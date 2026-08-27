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
    with patch("app.graph.nodes.generation.synthesize.generate_content_safe") as mock_generate:
        result = synthesize({"user_message": "test", "retrieved_chunks": []})

        assert result["confidence"] == "Low"
        assert result["sources"] == []
        mock_generate.assert_not_called()


@patch("app.graph.nodes.generation.synthesize.generate_content_safe")
def test_high_confidence_when_top_score_above_threshold(mock_generate):
    mock_generate.return_value = _fake_llm_response("Vikas Sir and Team Lead approve WFH.")

    chunks = [_fake_chunk(HIGH_CONFIDENCE_THRESHOLD + 0.01)]
    result = synthesize({"user_message": "who approves WFH", "retrieved_chunks": chunks})

    assert result["confidence"] == "High"


@patch("app.graph.nodes.generation.synthesize.generate_content_safe")
def test_medium_confidence_in_middle_range(mock_generate):
    mock_generate.return_value = _fake_llm_response("answer")

    chunks = [_fake_chunk((HIGH_CONFIDENCE_THRESHOLD + MEDIUM_CONFIDENCE_THRESHOLD) / 2)]
    result = synthesize({"user_message": "test", "retrieved_chunks": chunks})

    assert result["confidence"] == "Medium"


@patch("app.graph.nodes.generation.synthesize.generate_content_safe")
def test_low_confidence_below_threshold(mock_generate):
    mock_generate.return_value = _fake_llm_response("answer")

    chunks = [_fake_chunk(MEDIUM_CONFIDENCE_THRESHOLD - 0.05)]
    result = synthesize({"user_message": "test", "retrieved_chunks": chunks})

    assert result["confidence"] == "Low"


@patch("app.graph.nodes.generation.synthesize.generate_content_safe")
def test_sources_include_document_and_heading(mock_generate):
    mock_generate.return_value = _fake_llm_response("answer")

    chunks = [_fake_chunk(0.7, title="WFH Policy", heading="General WFH Policy - FAQ")]
    result = synthesize({"user_message": "test", "retrieved_chunks": chunks})

    assert result["sources"][0]["document_title"] == "WFH Policy"
    assert result["sources"][0]["heading_path"] == "General WFH Policy - FAQ"


@patch("app.graph.nodes.generation.synthesize.generate_content_safe")
def test_uses_resolved_query_over_raw_message_when_present(mock_generate):
    mock_generate.return_value = _fake_llm_response("answer")

    chunks = [_fake_chunk(0.7)]
    synthesize({
        "user_message": "what about for interns",
        "resolved_query": "What is the WFH policy for interns?",
        "retrieved_chunks": chunks,
    })

    _, kwargs = mock_generate.call_args
    assert "What is the WFH policy for interns?" in kwargs["contents"]


@patch("app.graph.nodes.generation.synthesize.generate_content_safe")
def test_answer_comes_from_llm_response_text(mock_generate):
    mock_generate.return_value = _fake_llm_response(
        "Both the Team Lead and Vikas Sir must approve."
    )

    chunks = [_fake_chunk(0.7)]
    result = synthesize({"user_message": "test", "retrieved_chunks": chunks})

    assert result["answer"] == "Both the Team Lead and Vikas Sir must approve."


@patch("app.graph.nodes.generation.synthesize.generate_content_safe")
def test_duplicate_sources_are_deduplicated(mock_generate):
    mock_generate.return_value = _fake_llm_response("answer")

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

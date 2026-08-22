import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.graph.nodes.retrieval.retrieve import retrieve
from app.retrieval.retrievers.hybrid_retriever import RetrievedChunk


def _fake_chunk(title="Doc"):
    return RetrievedChunk(
        chunk_id="1", document_id="1", document_title=title, document_version="1.0",
        heading_path="H", content="c", category=None, sub_category=None,
        people=[], roles=[], fine_grained_topics=[],
        vector_score=0.5, text_score=0.5, combined_score=0.5,
    )


@patch("app.graph.nodes.retrieval.retrieve.hybrid_search")
def test_uses_user_message_when_no_resolved_query(mock_search):
    mock_search.return_value = [_fake_chunk()]

    retrieve({"user_message": "what is the WFH policy"})

    args, kwargs = mock_search.call_args
    assert args[0] == "what is the WFH policy"


@patch("app.graph.nodes.retrieval.retrieve.hybrid_search")
def test_uses_resolved_query_when_present(mock_search):
    """When resolve_follow_up has already run, its resolved_query must
    be used instead of the raw (context-dependent) user_message."""
    mock_search.return_value = [_fake_chunk()]

    retrieve({
        "user_message": "what about for interns",
        "resolved_query": "what is the WFH policy for interns",
    })

    args, kwargs = mock_search.call_args
    assert args[0] == "what is the WFH policy for interns"


@patch("app.graph.nodes.retrieval.retrieve.hybrid_search")
def test_returns_chunks_in_state(mock_search):
    chunks = [_fake_chunk("A"), _fake_chunk("B")]
    mock_search.return_value = chunks

    result = retrieve({"user_message": "test"})

    assert result == {"retrieved_chunks": chunks}


@patch("app.graph.nodes.retrieval.retrieve.hybrid_search")
def test_requests_top_20(mock_search):
    mock_search.return_value = []

    retrieve({"user_message": "test"})

    _, kwargs = mock_search.call_args
    assert kwargs["limit"] == 20

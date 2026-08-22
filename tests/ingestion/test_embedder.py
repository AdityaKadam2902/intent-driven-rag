"""
Mocked tests - check that our code calls Gemini's embedding API
correctly, without needing a real API key. See
scripts/test_embedding_live.py for the real, live version.
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.ingestion.embeddings.embedder import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    embed_chunks,
    embed_document_chunk,
    embed_query,
)


def _fake_embedding_response(values):
    fake_embedding = MagicMock()
    fake_embedding.values = values
    fake_response = MagicMock()
    fake_response.embeddings = [fake_embedding]
    return fake_response


@patch("app.ingestion.embeddings.embedder.get_client")
def test_embed_document_chunk_uses_retrieval_document_task_type(mock_get_client):
    """The single most important behavior in this file: storing a chunk
    must use RETRIEVAL_DOCUMENT, not RETRIEVAL_QUERY - using the wrong
    one doesn't error, it just quietly hurts retrieval quality later,
    which is exactly why this needs a hard regression test."""
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = _fake_embedding_response([0.1, 0.2, 0.3])
    mock_get_client.return_value = mock_client

    embed_document_chunk("some SOP text")

    _, kwargs = mock_client.models.embed_content.call_args
    assert kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"


@patch("app.ingestion.embeddings.embedder.get_client")
def test_embed_query_uses_retrieval_query_task_type(mock_get_client):
    """The other half of the same critical distinction - a user's
    question must use RETRIEVAL_QUERY, not RETRIEVAL_DOCUMENT."""
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = _fake_embedding_response([0.1, 0.2, 0.3])
    mock_get_client.return_value = mock_client

    embed_query("what is the WFH policy")

    _, kwargs = mock_client.models.embed_content.call_args
    assert kwargs["config"].task_type == "RETRIEVAL_QUERY"


@patch("app.ingestion.embeddings.embedder.get_client")
def test_uses_stable_model_not_preview(mock_get_client):
    """Regression test for the deliberate model choice: must be
    gemini-embedding-001 (GA/stable), not gemini-embedding-2
    (preview, multimodal, incompatible embedding space)."""
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = _fake_embedding_response([0.1])
    mock_get_client.return_value = mock_client

    embed_document_chunk("text")

    _, kwargs = mock_client.models.embed_content.call_args
    assert kwargs["model"] == "gemini-embedding-001"
    assert EMBEDDING_MODEL == "gemini-embedding-001"


@patch("app.ingestion.embeddings.embedder.get_client")
def test_embed_document_chunk_returns_the_vector(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = _fake_embedding_response([0.1, 0.2, 0.3])
    mock_get_client.return_value = mock_client

    vector = embed_document_chunk("some text")

    assert vector == [0.1, 0.2, 0.3]


@patch("app.ingestion.embeddings.embedder.get_client")
def test_output_dimensionality_is_set(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = _fake_embedding_response([0.1])
    mock_get_client.return_value = mock_client

    embed_document_chunk("text")

    _, kwargs = mock_client.models.embed_content.call_args
    assert kwargs["config"].output_dimensionality == EMBEDDING_DIMENSIONS


@patch("app.ingestion.embeddings.embedder.get_client")
@patch("app.ingestion.embeddings.embedder.time.sleep")
def test_embed_chunks_paces_calls_but_not_before_the_first(mock_sleep, mock_get_client):
    from app.ingestion.processors.chunker import Chunk

    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = _fake_embedding_response([0.1])
    mock_get_client.return_value = mock_client

    chunks = [
        Chunk(heading_path="A", content="a", chunk_index=0),
        Chunk(heading_path="B", content="b", chunk_index=1),
        Chunk(heading_path="C", content="c", chunk_index=2),
    ]

    results = embed_chunks(chunks)

    assert mock_sleep.call_count == 2
    assert len(results) == 3
    assert results[0] == [0.1]

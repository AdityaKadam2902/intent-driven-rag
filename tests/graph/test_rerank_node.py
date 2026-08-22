import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.graph.nodes.retrieval.rerank import RERANK_TOP_N, rerank
from app.retrieval.retrievers.hybrid_retriever import RetrievedChunk


def _fake_chunk(idx):
    return RetrievedChunk(
        chunk_id=str(idx), document_id="1", document_title="Doc", document_version="1.0",
        heading_path=f"H{idx}", content=f"c{idx}", category=None, sub_category=None,
        people=[], roles=[], fine_grained_topics=[],
        vector_score=0.5, text_score=0.5, combined_score=0.5,
    )


def test_narrows_20_to_5():
    chunks = [_fake_chunk(i) for i in range(20)]

    result = rerank({"retrieved_chunks": chunks})

    assert len(result["retrieved_chunks"]) == RERANK_TOP_N
    assert RERANK_TOP_N == 5


def test_preserves_order():
    """Since hybrid_search already sorts by combined_score, rerank must
    not shuffle the order - it only slices."""
    chunks = [_fake_chunk(i) for i in range(10)]

    result = rerank({"retrieved_chunks": chunks})

    assert result["retrieved_chunks"] == chunks[:5]


def test_handles_fewer_than_5_chunks():
    chunks = [_fake_chunk(0), _fake_chunk(1)]

    result = rerank({"retrieved_chunks": chunks})

    assert len(result["retrieved_chunks"]) == 2


def test_handles_zero_chunks():
    result = rerank({"retrieved_chunks": []})

    assert result["retrieved_chunks"] == []


def test_handles_missing_key_gracefully():
    result = rerank({})

    assert result["retrieved_chunks"] == []

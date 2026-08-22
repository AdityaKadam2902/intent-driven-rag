"""
Embedding generation - turns chunk text into vectors for semantic search.

Uses gemini-embedding-001 specifically, not gemini-embedding-2:
  - gemini-embedding-001 is GA (stable, general availability).
  - gemini-embedding-2 still carries a "-preview" suffix as of this
    writing, meaning its API/pricing could still change - and we've
    already been burned once this project by a preview/deprecated
    model being pulled early (gemini-2.5-flash, June 2026).
  - gemini-embedding-2 is multimodal (text/image/video/audio/PDF) - we
    have no multimodal need, SOPs are plain text only, so we'd be
    paying 33% more for capability we don't use.
  - Critically: the two models' embedding spaces are INCOMPATIBLE. If
    we ever did want to switch later, every existing embedding would
    need to be regenerated from scratch anyway - so there's no cost to
    starting on -001 now and no easy "upgrade path" to worry about
    losing.

Task-type aware: RETRIEVAL_DOCUMENT for what gets stored (chunks),
RETRIEVAL_QUERY for what a user asks at query time. Using the same task
type for both measurably hurts retrieval quality - this is a documented,
easy-to-miss mistake, called out explicitly in the architecture docs.
"""
import time

from google.genai import types

from app.llm.gemini import get_client

EMBEDDING_MODEL = "gemini-embedding-001"

# 768 keeps storage cost down (pgvector index size, row size) with only
# a small, documented quality trade-off vs the max 3072 - a reasonable
# default for SOP-scale data. Revisit if retrieval quality testing
# later shows it's not enough.
EMBEDDING_DIMENSIONS = 768


def embed_document_chunk(text: str) -> list[float]:
    """Embed a chunk of SOP content for STORAGE. Must use
    RETRIEVAL_DOCUMENT - using RETRIEVAL_QUERY here would hurt search
    quality even though the API call would still succeed without
    error, which is exactly what makes this mistake easy to miss."""
    client = get_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    return response.embeddings[0].values


def embed_query(text: str) -> list[float]:
    """Embed a user's question at query time. Must use RETRIEVAL_QUERY,
    not RETRIEVAL_DOCUMENT - see module docstring."""
    client = get_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    return response.embeddings[0].values


def embed_chunks(chunks) -> dict[int, list[float]]:
    """Embed every chunk of a document, keyed by chunk_index. Paced
    with a short delay to stay under free-tier rate limits, same
    lesson learned from fact extraction and entity tagging - see
    https://ai.google.dev/gemini-api/docs/rate-limits.
    This pacing is for local/manual testing only; production ingestion
    should use the Batch API instead of one-at-a-time calls (see
    module docstring reference to the Batch API being cheaper too)."""
    results = {}
    for i, chunk in enumerate(chunks):
        if i > 0:
            time.sleep(4.5)
        results[chunk.chunk_index] = embed_document_chunk(chunk.content)
    return results

"""
Run this manually, from the project root, with a real GEMINI_API_KEY set
in your .env file:

    python scripts/test_embedding_live.py

Makes REAL calls to the Gemini embeddings API (paced to stay under the
free-tier rate limit).

What to check when you run this:
- Does every chunk get a vector of the expected length (768 numbers)?
- Sanity check on task-type mattering: this script also embeds a query
  ("who approves WFH requests") using RETRIEVAL_QUERY, and compares it
  against the WFH approval chunk embedded with RETRIEVAL_DOCUMENT - the
  similarity should be meaningfully high, since that's the actual
  question this chunk answers. This is the first real evidence that
  semantic search will work at all once storage exists.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.loaders.txt_loader import TxtLoader
from app.ingestion.processors.chunker import chunk_sop
from app.ingestion.embeddings.embedder import (
    EMBEDDING_DIMENSIONS,
    embed_chunks,
    embed_query,
)

SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples"))


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(y * y for y in b) ** 0.5
    return dot / (mag_a * mag_b)


def run_for_file(filename: str):
    loader = TxtLoader()
    doc = loader.load(os.path.join(SAMPLES_DIR, filename))
    chunks = chunk_sop(doc.body, doc.metadata.get("Title", ""))

    print(f"\n{'=' * 70}")
    print(f"FILE: {filename}  ({len(chunks)} chunks)")
    print(f"{'=' * 70}")

    embeddings = embed_chunks(chunks)

    for chunk_idx, vector in embeddings.items():
        chunk = next(c for c in chunks if c.chunk_index == chunk_idx)
        assert len(vector) == EMBEDDING_DIMENSIONS, (
            f"Expected {EMBEDDING_DIMENSIONS} dimensions, got {len(vector)}"
        )
        print(f"[chunk {chunk_idx}] ({chunk.heading_path}): {len(vector)} dims OK")

    return chunks, embeddings


if __name__ == "__main__":
    wfh_chunks, wfh_embeddings = run_for_file("wfh_policy.txt")
    run_for_file("testing_process.txt")
    run_for_file("sdlc_module.txt")

    print(f"\n{'=' * 70}")
    print("SEMANTIC SEARCH SANITY CHECK")
    print(f"{'=' * 70}")
    print("Embedding a real question and comparing against every WFH chunk...")

    query_vector = embed_query("who approves WFH requests")

    scored = []
    for chunk in wfh_chunks:
        vec = wfh_embeddings[chunk.chunk_index]
        score = cosine_similarity(query_vector, vec)
        scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])

    print("\nTop 3 matches for \"who approves WFH requests\":")
    for score, chunk in scored[:3]:
        preview = chunk.content.replace("\n", " | ")[:100]
        print(f"  score={score:.4f}  [{chunk.heading_path}]  {preview}")

    print("\nExpect the top match to be the chunk containing "
          "\"WFH requests must be reviewed by the Team Lead and approved by Vikas Sir\".")

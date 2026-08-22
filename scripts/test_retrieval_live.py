"""
Run this manually, from the project root, AFTER running
scripts/run_ingestion.py at least once (so there's real data to search).
Requires your real GEMINI_API_KEY and Postgres connection in .env.

    python scripts/test_retrieval_live.py

This is the real end-to-end proof: your actual ingested SOP chunks,
searched with your actual Gemini embeddings, through the actual hybrid
SQL query - not a mock, not a fixture.

What to check: for "who approves WFH requests", does the top result
contain "Vikas Sir" and "Team Lead"? This is the same question and
expected answer already proven at the embedding-only level in
scripts/test_embedding_live.py - this script proves the FULL retrieval
path (embedding + keyword + Postgres query + filtering) gives the same
correct answer, not just the raw embedding similarity in isolation.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retrieval.retrievers.hybrid_retriever import hybrid_search


def run_query(question: str, limit: int = 5):
    print(f"\n{'=' * 70}")
    print(f"QUERY: {question}")
    print(f"{'=' * 70}")

    results = hybrid_search(question, limit=limit)

    if not results:
        print("No results. Have you run scripts/run_ingestion.py yet?")
        return

    for i, r in enumerate(results):
        preview = r.content.replace("\n", " | ")[:110]
        print(f"\n[{i+1}] combined={r.combined_score:.4f} "
              f"(vector={r.vector_score:.4f}, text={r.text_score:.4f})")
        print(f"    doc: {r.document_title} v{r.document_version}")
        print(f"    heading: {r.heading_path}")
        print(f"    content: {preview}")
        if r.people or r.roles:
            print(f"    people={r.people}  roles={r.roles}")


if __name__ == "__main__":
    run_query("who approves WFH requests")
    run_query("what happens if an intern's leave runs out")
    run_query("how are test cases created")
    run_query("what tools are used for defect tracking")

"""
This makes REAL calls to the Gemini API (small cost, a handful of calls).
It is not part of the automated test suite (tests/) because those must
run without needing real credentials or network access.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.loaders.txt_loader import TxtLoader
from app.ingestion.processors.chunker import chunk_sop
from app.ingestion.processors.fact_extractor import extract_facts_for_chunks

SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples"))


def run_for_file(filename: str):
    loader = TxtLoader()
    doc = loader.load(os.path.join(SAMPLES_DIR, filename))
    chunks = chunk_sop(doc.body, doc.metadata.get("Title", ""))

    print(f"\n{'=' * 70}")
    print(f"FILE: {filename}  ({len(chunks)} chunks)")
    print(f"{'=' * 70}")

    results = extract_facts_for_chunks(chunks, doc_title=doc.metadata.get("Title", ""))

    if not results:
        print("No facts extracted from any chunk.")
        return

    for chunk_idx, facts in results.items():
        chunk = next(c for c in chunks if c.chunk_index == chunk_idx)
        print(f"\n[chunk {chunk_idx}] ({chunk.heading_path}):")
        for f in facts:
            print(f"  - policy: {f.policy}")
            print(f"    value: {f.value}")
            if f.condition:
                print(f"    condition: {f.condition}")


if __name__ == "__main__":
    for f in ["wfh_policy.txt", "testing_process.txt", "sdlc_module.txt"]:
        run_for_file(f)

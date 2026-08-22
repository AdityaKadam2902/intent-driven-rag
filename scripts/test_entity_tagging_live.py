"""
Run this manually, from the project root, with a real GEMINI_API_KEY set
in your .env file:

    python scripts/test_entity_tagging_live.py

Makes REAL calls to the Gemini API (paced to stay under the free-tier
rate limit - this will take a little while to run, that's expected).

What to check when you run this:
- Does "Vikas Sir" get tagged as a PERSON and "Team Lead" as a ROLE,
  correctly separated?
- Is casing now consistent (e.g. always "Intern", never a mix of
  "intern" and "Intern" across chunks)?
- Do fine_grained_topics only ever show values from the fixed list in
  entity_tagger.py - never an invented one-off label?
- Does every chunk correctly show category/sub_category from the
  document's own metadata (e.g. "HR / WFH" for every chunk in the WFH
  file), since that no longer depends on the AI at all?
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.loaders.txt_loader import TxtLoader
from app.ingestion.processors.chunker import chunk_sop
from app.ingestion.processors.entity_tagger import tag_entities_for_chunks

SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples"))


def run_for_file(filename: str):
    loader = TxtLoader()
    doc = loader.load(os.path.join(SAMPLES_DIR, filename))
    chunks = chunk_sop(doc.body, doc.metadata.get("Title", ""))

    category = doc.metadata.get("Category")
    sub_category = doc.metadata.get("Sub Category")

    print(f"\n{'=' * 70}")
    print(f"FILE: {filename}  ({len(chunks)} chunks)")
    print(f"Document-level tags (from metadata, same for every chunk): "
          f"{category} / {sub_category}")
    print(f"{'=' * 70}")

    results = tag_entities_for_chunks(
        chunks, doc_title=doc.metadata.get("Title", ""),
        category=category, sub_category=sub_category,
    )

    for chunk_idx, tags in results.items():
        chunk = next(c for c in chunks if c.chunk_index == chunk_idx)
        if not tags.people and not tags.roles and not tags.fine_grained_topics:
            continue  # skip printing chunks with nothing tagged beyond category
        print(f"\n[chunk {chunk_idx}] ({chunk.heading_path}):")
        if tags.people:
            print(f"  people: {tags.people}")
        if tags.roles:
            print(f"  roles: {tags.roles}")
        if tags.fine_grained_topics:
            print(f"  fine_grained_topics: {tags.fine_grained_topics}")


if __name__ == "__main__":
    for f in ["wfh_policy.txt", "testing_process.txt", "sdlc_module.txt"]:
        run_for_file(f)

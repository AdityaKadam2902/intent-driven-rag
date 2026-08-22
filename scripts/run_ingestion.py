"""
Run this manually, from the project root, with a real GEMINI_API_KEY and
a running Postgres (with schema.sql applied) configured in your .env:

    python scripts/run_ingestion.py

This is the REAL pipeline - processes every .txt file in samples/,
however many there are, not a hardcoded list. Run it twice in a row and
the second run should skip everything (version unchanged) - that's the
version-check working correctly, not a bug.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.pipeline import ingest_directory

SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "samples"))

if __name__ == "__main__":
    print(f"Ingesting all .txt files from: {SAMPLES_DIR}\n")
    summary = ingest_directory(SAMPLES_DIR)

    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"Inserted (new documents): {summary['inserted']}")
    print(f"Updated (version changed): {summary['updated']}")
    print(f"Skipped (unchanged): {summary['skipped_unchanged']}")
    if summary["errors"]:
        print(f"Errors: {len(summary['errors'])}")
        for e in summary["errors"]:
            print(f"  - {e}")

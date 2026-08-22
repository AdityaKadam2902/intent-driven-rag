"""
The real ingestion pipeline - processes every SOP file in a directory
through the full chain: load -> chunk -> tag entities -> embed -> store.

This is the generic orchestrator that test_fact_extraction_live.py and
similar scripts were explicitly NOT meant to be (see earlier discussion
in project history - those are one-off manual verification scripts for
individual pieces; this is the real thing that will process however
many SOP files actually exist, not a hardcoded list).

Deliberately SKIPS fact extraction - see
docs/decisions/001-pause-fact-extraction.md. Nothing here calls
extract_facts_for_chunks(). Resuming fact extraction later means adding
one step here, not rebuilding anything.

Version-aware: uses get_document_version() to skip re-processing a
document whose version hasn't changed, since chunking + entity tagging
+ embedding is real API cost we don't want to pay for no-op re-ingests.
"""
from app.ingestion.embeddings.embedder import embed_chunks
from app.ingestion.loaders.txt_loader import TxtLoader
from app.ingestion.processors.chunker import chunk_sop
from app.ingestion.processors.entity_tagger import tag_entities_for_chunks
from app.ingestion.processors.structure_validator import assess_structure_quality
from app.repositories.document_repository import (
    ChunkRecord,
    get_document_version,
    upsert_document,
)


def ingest_directory(source_dir: str, verbose: bool = True) -> dict:
    """Processes every .txt file in source_dir. Returns a summary dict
    with counts of documents inserted/updated/skipped, for visibility
    into what actually happened on a given run."""
    loader = TxtLoader()
    docs = loader.load_all(source_dir)

    summary = {
        "inserted": 0, "updated": 0, "skipped_unchanged": 0,
        "flagged_structure": 0, "errors": [],
    }

    for doc in docs:
        title = doc.metadata.get("Title")
        incoming_version = doc.metadata.get("Version")

        if not title or not incoming_version:
            summary["errors"].append(f"{doc.source_ref}: missing Title or Version in metadata")
            continue

        try:
            existing_version = get_document_version(title)

            if existing_version == incoming_version:
                summary["skipped_unchanged"] += 1
                if verbose:
                    print(f"SKIP (unchanged, v{incoming_version}): {title}")
                continue

            # Check structure quality BEFORE chunking - see
            # app/ingestion/processors/structure_validator.py. This is
            # a deliberate gate: a flattened/messy file gets flagged
            # and skipped rather than silently chunked badly, which
            # would produce unreliable heading boundaries and later
            # cause wrong or incomplete answers.
            quality = assess_structure_quality(doc.body)
            if not quality.is_reliable:
                summary["flagged_structure"] += 1
                if verbose:
                    print(f"FLAGGED (structure quality, v{incoming_version}): {title}")
                    for w in quality.warnings:
                        print(f"    - {w}")
                continue

            chunks = chunk_sop(doc.body, doc_title=title)

            category = doc.metadata.get("Category")
            sub_category = doc.metadata.get("Sub Category")

            tags_by_index = tag_entities_for_chunks(
                chunks, doc_title=title, category=category, sub_category=sub_category
            )
            embeddings_by_index = embed_chunks(chunks)

            chunk_records = []
            for chunk in chunks:
                tags = tags_by_index.get(chunk.chunk_index)
                embedding = embeddings_by_index.get(chunk.chunk_index)
                chunk_records.append(
                    ChunkRecord(
                        chunk_index=chunk.chunk_index,
                        heading_path=chunk.heading_path,
                        content=chunk.content,
                        category=tags.category if tags else category,
                        sub_category=tags.sub_category if tags else sub_category,
                        people=tags.people if tags else [],
                        roles=tags.roles if tags else [],
                        fine_grained_topics=tags.fine_grained_topics if tags else [],
                        embedding=embedding,
                    )
                )

            metadata_with_source = {**doc.metadata, "source_ref": doc.source_ref}
            upsert_document(metadata_with_source, chunk_records)

            if existing_version is None:
                summary["inserted"] += 1
                if verbose:
                    print(f"INSERT (new, v{incoming_version}): {title} - {len(chunks)} chunks")
            else:
                summary["updated"] += 1
                if verbose:
                    print(
                        f"UPDATE (v{existing_version} -> v{incoming_version}): "
                        f"{title} - {len(chunks)} chunks"
                    )

        except Exception as e:
            summary["errors"].append(f"{title}: {e}")
            if verbose:
                print(f"ERROR: {title}: {e}")

    return summary

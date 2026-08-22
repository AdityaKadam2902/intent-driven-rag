"""
Document repository - the ONLY place that writes to sop_documents and
sop_chunks. Nodes/pipeline code call these functions, never raw SQL
directly - this is what lets the data-delivery mechanism (still
undecided by Meldep's team) change later without touching anything
that calls this repository.

This is also where the answer to "what happens to old facts/tags when
a document changes" actually gets implemented: upsert_document() is
atomic - new chunks are fully written before old ones for that
document are deleted, all in one transaction. There is never a moment
where a query could see a half-updated document, and nothing from a
previous version lingers after a successful update.
"""
from dataclasses import dataclass

from app.database.postgres import get_connection


@dataclass
class ChunkRecord:
    chunk_index: int
    heading_path: str
    content: str
    category: str | None
    sub_category: str | None
    people: list[str]
    roles: list[str]
    fine_grained_topics: list[str]
    embedding: list[float]


def get_document_version(title: str) -> str | None:
    """Returns the currently stored version for a document title, or
    None if it doesn't exist yet. Used by the ingestion pipeline to
    decide: skip (same version), update (newer version), or insert
    (new document) - see app/ingestion/pipeline.py (not yet built)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT version FROM sop_documents WHERE title = %s "
                "ORDER BY ingested_at DESC LIMIT 1",
                (title,),
            )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def upsert_document(
    metadata: dict,
    chunks: list[ChunkRecord],
) -> str:
    """Atomically writes a document and its chunks. If a document with
    the same title already exists, its old chunks are deleted as part
    of the SAME transaction that inserts the new ones - so a query
    running concurrently either sees the complete old version or the
    complete new version, never a mix, and never a moment with no
    chunks at all for a document that should have them.

    Returns the new document_id (as a string).
    """
    conn = get_connection()
    try:
        with conn:  # transaction: commits on success, rolls back on any exception
            with conn.cursor() as cur:
                # Find existing document (if any) by title, to know
                # whether this is an update (delete old chunks after
                # inserting new ones) or a brand new document.
                cur.execute(
                    "SELECT id FROM sop_documents WHERE title = %s "
                    "ORDER BY ingested_at DESC LIMIT 1",
                    (metadata.get("Title"),),
                )
                existing = cur.fetchone()
                old_document_id = existing[0] if existing else None

                # Insert the new document row (new version).
                cur.execute(
                    """
                    INSERT INTO sop_documents
                        (title, category, sub_category, version, status,
                         is_active, created_by, created_date, updated_by,
                         updated_date, source_ref)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (title, version) DO UPDATE
                        SET status = EXCLUDED.status,
                            is_active = EXCLUDED.is_active
                    RETURNING id
                    """,
                    (
                        metadata.get("Title"),
                        metadata.get("Category"),
                        metadata.get("Sub Category"),
                        metadata.get("Version"),
                        metadata.get("Status"),
                        metadata.get("Is Active"),
                        metadata.get("Created By"),
                        metadata.get("Created Date"),
                        metadata.get("Updated By"),
                        metadata.get("Updated Date"),
                        metadata.get("source_ref"),
                    ),
                )
                new_document_id = cur.fetchone()[0]

                # Insert every new chunk for this document/version.
                for c in chunks:
                    cur.execute(
                        """
                        INSERT INTO sop_chunks
                            (document_id, document_version, chunk_index,
                             heading_path, content, category, sub_category,
                             people, roles, fine_grained_topics, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            new_document_id,
                            metadata.get("Version"),
                            c.chunk_index,
                            c.heading_path,
                            c.content,
                            c.category,
                            c.sub_category,
                            c.people,
                            c.roles,
                            c.fine_grained_topics,
                            c.embedding,
                        ),
                    )

                # Only now, after the new chunks are fully written in
                # this same transaction, remove the OLD document row
                # (if this was an update, not a brand new document).
                # ON DELETE CASCADE removes its old chunks automatically.
                # This is the atomic swap: if anything above failed, this
                # line never runs and the whole transaction rolls back -
                # the old version stays fully intact and queryable.
                if old_document_id and old_document_id != new_document_id:
                    cur.execute(
                        "DELETE FROM sop_documents WHERE id = %s", (old_document_id,)
                    )

                return str(new_document_id)
    finally:
        conn.close()


def get_servable_chunk_count() -> int:
    """Returns how many chunks currently exist for Published + Active
    documents - a simple sanity-check query for tests and manual
    verification, not used by the real pipeline."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM sop_chunks c
                JOIN sop_documents d ON d.id = c.document_id
                WHERE d.status = 'Published' AND d.is_active = 'Active'
                """
            )
            return cur.fetchone()[0]
    finally:
        conn.close()


def get_ingested_documents() -> list[dict]:
    """Returns one row per currently-stored document: title, category,
    sub_category, version, status, is_active, chunk_count, ingested_at.
    Used by the web app's document management page."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.title, d.category, d.sub_category, d.version,
                       d.status, d.is_active, d.ingested_at, COUNT(c.id) AS chunk_count
                FROM sop_documents d
                LEFT JOIN sop_chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.ingested_at DESC
                """
            )
            rows = cur.fetchall()
            return [
                {
                    "title": r[0],
                    "category": r[1],
                    "sub_category": r[2],
                    "version": r[3],
                    "status": r[4],
                    "is_active": r[5],
                    "ingested_at": r[6],
                    "chunk_count": r[7],
                }
                for r in rows
            ]
    finally:
        conn.close()

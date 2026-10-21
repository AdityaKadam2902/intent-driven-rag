"""
Read-only helper for the cache layer: which documents are currently
servable, and at what version.

Lives in its own module (not document_repository.py) on purpose -
document_repository.py is the single writer of sop_documents/sop_chunks
and is also extended by the web app, so the cache layer's read needs
don't touch it.

The returned {title: version} dict is part of every cache key. Any
document update, publish, or deactivation changes this dict, which
changes the key, so stale entries are simply never looked up again.
This is deliberately coarse (one document changing invalidates every
cached answer) - correct and simple first; per-document tagging can
come later if the hit rate ever suffers from it.
"""
from app.database.postgres import get_connection, with_db_breaker


def _query_versions():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT title, version FROM sop_documents "
                "WHERE status = 'Published' AND is_active = 'Active'"
            )
            return cur.fetchall()
    finally:
        conn.close()


def get_all_document_versions() -> dict[str, str]:
    """Returns {title: version} for every Published + Active document,
    through the shared Postgres circuit breaker."""
    return {title: version for title, version in with_db_breaker(_query_versions)}

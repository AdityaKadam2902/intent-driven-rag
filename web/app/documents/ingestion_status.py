"""
Owns the ingestion_runs table. Separate from document_repository.py,
which owns sop_documents/sop_chunks - this table only tracks the
web layer's background-job status, not RAG data itself.
"""
from web.app.db import get_connection


def start_run(assistant_id: str) -> int:
    with get_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO ingestion_runs (assistant_id, status)
            VALUES (%s, 'running')
            RETURNING id
            """,
            (assistant_id,),
        ).fetchone()
        return row["id"]


def mark_success(run_id: int, doc_count: int) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE ingestion_runs
            SET status = 'success', doc_count = %s, finished_at = now()
            WHERE id = %s
            """,
            (doc_count, run_id),
        )


def mark_failed(run_id: int, error_message: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE ingestion_runs
            SET status = 'failed', error_message = %s, finished_at = now()
            WHERE id = %s
            """,
            (error_message, run_id),
        )


def get_latest_run(assistant_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, status, doc_count, error_message, started_at, finished_at
            FROM ingestion_runs
            WHERE assistant_id = %s
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (assistant_id,),
        ).fetchone()
        return dict(row) if row else None

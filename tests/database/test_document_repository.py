"""
These are INTEGRATION tests - they run against a real local Postgres
database, not mocks. This is deliberate: transaction behavior, foreign
key cascades, and array/vector column types are exactly the kind of
thing a mock would hide bugs in. Requires a running Postgres with the
schema applied - see README for setup, or run:

    psql -d meldep_sop_dev -f app/database/schema.sql

Each test cleans up after itself so tests can run repeatedly.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
from app.repositories.document_repository import (
    ChunkRecord,
    get_document_version,
    get_ingested_documents,
    get_servable_chunk_count,
    upsert_document,
)
from app.database.postgres import get_connection
from app.repositories.document_repository import (
    ChunkRecord,
    get_document_version,
    get_servable_chunk_count,
    upsert_document,
)


def _cleanup(title: str):
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM sop_documents WHERE title = %s", (title,))
    finally:
        conn.close()


@pytest.fixture
def sample_chunks():
    return [
        ChunkRecord(
            chunk_index=0,
            heading_path="Test Heading",
            content="Test content for chunk 0",
            category="HR",
            sub_category="WFH",
            people=["Vikas Sir"],
            roles=["Team Lead"],
            fine_grained_topics=["approval"],
            embedding=[0.1] * 768,
        ),
        ChunkRecord(
            chunk_index=1,
            heading_path="Test Heading 2",
            content="Test content for chunk 1",
            category="HR",
            sub_category="WFH",
            people=[],
            roles=["Intern"],
            fine_grained_topics=["eligibility"],
            embedding=[0.2] * 768,
        ),
    ]


def test_insert_new_document_and_chunks(sample_chunks):
    title = "TEST_DOC_insert_new"
    _cleanup(title)
    try:
        metadata = {
            "Title": title, "Category": "HR", "Sub Category": "WFH",
            "Version": "1.0", "Status": "Published", "Is Active": "Active",
        }
        doc_id = upsert_document(metadata, sample_chunks)
        assert doc_id is not None
        assert get_document_version(title) == "1.0"
    finally:
        _cleanup(title)


def test_update_swaps_old_chunks_for_new_atomically(sample_chunks):
    """This is the actual test for the question that started this
    whole piece: when a document updates, do old facts/tags/chunks get
    replaced, not just added alongside? This proves yes."""
    title = "TEST_DOC_atomic_swap"
    _cleanup(title)
    try:
        metadata_v1 = {
            "Title": title, "Category": "HR", "Sub Category": "WFH",
            "Version": "1.0", "Status": "Published", "Is Active": "Active",
        }
        doc_id_v1 = upsert_document(metadata_v1, sample_chunks)

        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM sop_chunks WHERE document_id = %s", (doc_id_v1,))
            assert cur.fetchone()[0] == 2
        conn.close()

        # Now "update" - new version, DIFFERENT chunk content (as if
        # the old fact/tag was removed from the source document)
        new_chunks = [
            ChunkRecord(
                chunk_index=0,
                heading_path="Completely New Heading",
                content="This replaces the old content entirely",
                category="HR",
                sub_category="WFH",
                people=[],
                roles=["Employee"],
                fine_grained_topics=["disciplinary"],
                embedding=[0.3] * 768,
            ),
        ]
        metadata_v2 = {**metadata_v1, "Version": "2.0"}
        doc_id_v2 = upsert_document(metadata_v2, new_chunks)

        assert get_document_version(title) == "2.0"

        conn = get_connection()
        with conn.cursor() as cur:
            # old document row should be GONE
            cur.execute("SELECT COUNT(*) FROM sop_documents WHERE id = %s", (doc_id_v1,))
            assert cur.fetchone()[0] == 0

            # old chunks should be GONE (proves the "does the old fact
            # still exist" question - no, it doesn't)
            cur.execute("SELECT COUNT(*) FROM sop_chunks WHERE document_id = %s", (doc_id_v1,))
            assert cur.fetchone()[0] == 0

            # new chunks should exist, and ONLY the new ones (1, not 2)
            cur.execute("SELECT COUNT(*) FROM sop_chunks WHERE document_id = %s", (doc_id_v2,))
            assert cur.fetchone()[0] == 1

            cur.execute("SELECT content FROM sop_chunks WHERE document_id = %s", (doc_id_v2,))
            assert cur.fetchone()[0] == "This replaces the old content entirely"
        conn.close()
    finally:
        _cleanup(title)


def test_unpublished_documents_are_not_counted_as_servable(sample_chunks):
    title = "TEST_DOC_unpublished"
    _cleanup(title)
    try:
        before = get_servable_chunk_count()
        metadata = {
            "Title": title, "Category": "HR", "Sub Category": "WFH",
            "Version": "1.0", "Status": "Draft", "Is Active": "Active",
        }
        upsert_document(metadata, sample_chunks)
        after = get_servable_chunk_count()
        # Draft status - should NOT increase the servable count
        assert after == before
    finally:
        _cleanup(title)


def test_inactive_documents_are_not_counted_as_servable(sample_chunks):
    title = "TEST_DOC_inactive"
    _cleanup(title)
    try:
        before = get_servable_chunk_count()
        metadata = {
            "Title": title, "Category": "HR", "Sub Category": "WFH",
            "Version": "1.0", "Status": "Published", "Is Active": "Inactive",
        }
        upsert_document(metadata, sample_chunks)
        after = get_servable_chunk_count()
        assert after == before
    finally:
        _cleanup(title)


def test_published_and_active_documents_are_servable(sample_chunks):
    title = "TEST_DOC_servable"
    _cleanup(title)
    try:
        before = get_servable_chunk_count()
        metadata = {
            "Title": title, "Category": "HR", "Sub Category": "WFH",
            "Version": "1.0", "Status": "Published", "Is Active": "Active",
        }
        upsert_document(metadata, sample_chunks)
        after = get_servable_chunk_count()
        assert after == before + 2  # both sample chunks now servable
    finally:
        _cleanup(title)


def test_array_fields_round_trip_correctly(sample_chunks):
    """people/roles/fine_grained_topics are Postgres arrays - make sure
    they actually store and retrieve as real lists, not strings."""
    title = "TEST_DOC_arrays"
    _cleanup(title)
    try:
        metadata = {
            "Title": title, "Category": "HR", "Sub Category": "WFH",
            "Version": "1.0", "Status": "Published", "Is Active": "Active",
        }
        doc_id = upsert_document(metadata, sample_chunks)

        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT people, roles, fine_grained_topics FROM sop_chunks "
                "WHERE document_id = %s AND chunk_index = 0", (doc_id,)
            )
            people, roles, topics = cur.fetchone()
            assert people == ["Vikas Sir"]
            assert roles == ["Team Lead"]
            assert topics == ["approval"]
        conn.close()
    finally:
        _cleanup(title)


def test_get_ingested_documents_includes_new_document_with_chunk_count(sample_chunks):
    title = "TEST_DOC_list_basic"
    _cleanup(title)
    try:
        metadata = {
            "Title": title, "Category": "HR", "Sub Category": "WFH",
            "Version": "1.0", "Status": "Published", "Is Active": "Active",
        }
        upsert_document(metadata, sample_chunks)

        docs = get_ingested_documents()
        match = next((d for d in docs if d["title"] == title), None)

        assert match is not None
        assert match["category"] == "HR"
        assert match["sub_category"] == "WFH"
        assert match["version"] == "1.0"
        assert match["status"] == "Published"
        assert match["is_active"] == "Active"
        assert match["chunk_count"] == 2
        assert match["ingested_at"] is not None
    finally:
        _cleanup(title)


def test_get_ingested_documents_reflects_atomic_swap(sample_chunks):
    """After an update, the list should show only the new version's
    chunk_count - not the old chunk count, and not double-counted."""
    title = "TEST_DOC_list_swap"
    _cleanup(title)
    try:
        metadata_v1 = {
            "Title": title, "Category": "HR", "Sub Category": "WFH",
            "Version": "1.0", "Status": "Published", "Is Active": "Active",
        }
        upsert_document(metadata_v1, sample_chunks)

        new_chunks = [
            ChunkRecord(
                chunk_index=0,
                heading_path="New Heading",
                content="Replacement content",
                category="HR",
                sub_category="WFH",
                people=[],
                roles=["Employee"],
                fine_grained_topics=["disciplinary"],
                embedding=[0.3] * 768,
            ),
        ]
        metadata_v2 = {**metadata_v1, "Version": "2.0"}
        upsert_document(metadata_v2, new_chunks)

        docs = get_ingested_documents()
        matches = [d for d in docs if d["title"] == title]

        assert len(matches) == 1  # old row is gone, not duplicated
        assert matches[0]["version"] == "2.0"
        assert matches[0]["chunk_count"] == 1
    finally:
        _cleanup(title)


def test_get_ingested_documents_orders_most_recent_first(sample_chunks):
    title_a = "TEST_DOC_list_order_a"
    title_b = "TEST_DOC_list_order_b"
    _cleanup(title_a)
    _cleanup(title_b)
    try:
        metadata_a = {
            "Title": title_a, "Category": "HR", "Sub Category": "WFH",
            "Version": "1.0", "Status": "Published", "Is Active": "Active",
        }
        upsert_document(metadata_a, sample_chunks)

        metadata_b = {**metadata_a, "Title": title_b}
        upsert_document(metadata_b, sample_chunks)

        docs = get_ingested_documents()
        titles_in_order = [d["title"] for d in docs]

        # b was ingested after a, so it should appear first (most recent first)
        assert titles_in_order.index(title_b) < titles_in_order.index(title_a)
    finally:
        _cleanup(title_a)
        _cleanup(title_b)
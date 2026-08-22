"""
Integration test - real Postgres, but the query embedding itself is
mocked (a hand-crafted vector) so this test is self-contained and
doesn't require real Gemini calls. It inserts its own fixture chunks
directly, proving the SQL scoring/filtering logic is correct.

Two real bugs were found and fixed while building this test, both
worth keeping documented here since they're easy to reintroduce:

1. Vector direction vs magnitude: an early version used
   [0.05]*768 intending it to be "less similar" than [0.1]*768 to the
   query vector. Cosine similarity is direction-only - scaling a vector
   doesn't change its direction, so [0.05]*768 and [0.1]*768 actually
   have IDENTICAL cosine similarity (1.0) to the query. Fixed by using
   vectors that are genuinely different in direction: a partial-overlap
   vector (cosine ~0.707) and a genuinely orthogonal vector (cosine 0.0).

2. Test isolation: an early version queried with no filter, competing
   against the ENTIRE live sop_chunks table (including real ingested
   data from scripts/run_ingestion.py). A deliberately low-scoring test
   row could get pushed out of the result window once enough real data
   existed, making the test fragile in a way that had nothing to do
   with whether retrieval actually works. Fixed by using
   hybrid_search()'s _filter_document_title parameter to scope each
   test to only its own fixture data.

For a true end-to-end proof against your REAL ingested SOP data, see
scripts/test_retrieval_live.py.
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.database.postgres import get_connection
from app.retrieval.retrievers.hybrid_retriever import hybrid_search

TEST_TITLE = "TEST_RETRIEVAL_DOC"


def _cleanup():
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM sop_documents WHERE title = %s", (TEST_TITLE,))
    finally:
        conn.close()


def _insert_fixture(status="Published", is_active="Active"):
    """Inserts one document with 3 chunks, using vectors that are
    genuinely distinct in DIRECTION (not just magnitude) from the
    query vector [0.1]*768:
      - chunk 0: identical direction -> cosine similarity 1.0
      - chunk 1: partial overlap      -> cosine similarity ~0.707
      - chunk 2: orthogonal           -> cosine similarity 0.0
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sop_documents
                        (title, category, sub_category, version, status, is_active)
                    VALUES (%s, 'HR', 'WFH', '1.0', %s, %s)
                    RETURNING id
                    """,
                    (TEST_TITLE, status, is_active),
                )
                doc_id = cur.fetchone()[0]

                identical_vector = [0.1] * 768              # cosine 1.0
                partial_overlap_vector = [0.1] * 384 + [0.0] * 384  # cosine ~0.707
                orthogonal_vector = [0.1, -0.1] * 384        # cosine 0.0

                chunks = [
                    (0, "Approval", "WFH requests need approval from the Team Lead.", identical_vector),
                    (1, "Other", "This chunk mentions approval as a keyword only.", partial_overlap_vector),
                    (2, "Unrelated", "This is about something completely different.", orthogonal_vector),
                ]
                for idx, heading, content, vec in chunks:
                    cur.execute(
                        """
                        INSERT INTO sop_chunks
                            (document_id, document_version, chunk_index, heading_path,
                             content, category, sub_category, people, roles,
                             fine_grained_topics, embedding)
                        VALUES (%s, '1.0', %s, %s, %s, 'HR', 'WFH', '{}', '{}', '{}', %s)
                        """,
                        (doc_id, idx, heading, content, vec),
                    )
        return doc_id
    finally:
        conn.close()


@patch("app.retrieval.retrievers.hybrid_retriever.embed_query")
def test_hybrid_search_ranks_vector_match_highest(mock_embed_query):
    """Chunk 0 has an identical-direction vector to the query - should
    rank #1. Scoped to this test's own fixture via
    _filter_document_title, so it can never be affected by how much
    real data exists in the same table."""
    _cleanup()
    try:
        _insert_fixture()
        mock_embed_query.return_value = [0.1] * 768

        results = hybrid_search(
            "approval process", limit=10, _filter_document_title=TEST_TITLE
        )

        assert len(results) == 3
        assert results[0].heading_path == "Approval"
        assert abs(results[0].vector_score - 1.0) < 0.001
    finally:
        _cleanup()


@patch("app.retrieval.retrievers.hybrid_retriever.embed_query")
def test_hybrid_search_excludes_unpublished_documents(mock_embed_query):
    """The servability filter (Published + Active) must actually work -
    an unpublished document's chunks should never appear in results."""
    _cleanup()
    try:
        _insert_fixture(status="Draft")
        mock_embed_query.return_value = [0.1] * 768

        results = hybrid_search(
            "approval process", limit=10, _filter_document_title=TEST_TITLE
        )

        assert len(results) == 0
    finally:
        _cleanup()


@patch("app.retrieval.retrievers.hybrid_retriever.embed_query")
def test_hybrid_search_excludes_inactive_documents(mock_embed_query):
    _cleanup()
    try:
        _insert_fixture(is_active="Inactive")
        mock_embed_query.return_value = [0.1] * 768

        results = hybrid_search(
            "approval process", limit=10, _filter_document_title=TEST_TITLE
        )

        assert len(results) == 0
    finally:
        _cleanup()


@patch("app.retrieval.retrievers.hybrid_retriever.embed_query")
def test_hybrid_search_respects_limit(mock_embed_query):
    _cleanup()
    try:
        _insert_fixture()
        mock_embed_query.return_value = [0.1] * 768

        results = hybrid_search(
            "approval process", limit=2, _filter_document_title=TEST_TITLE
        )

        assert len(results) == 2
    finally:
        _cleanup()


@patch("app.retrieval.retrievers.hybrid_retriever.embed_query")
def test_hybrid_search_returns_entity_tags_for_downstream_use(mock_embed_query):
    """Retrieved chunks must carry their category/roles/topics through -
    needed later for entity-filtered search and citations."""
    _cleanup()
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO sop_documents (title, category, sub_category, version, status, is_active) "
                    "VALUES (%s, 'HR', 'WFH', '1.0', 'Published', 'Active') RETURNING id",
                    (TEST_TITLE,),
                )
                doc_id = cur.fetchone()[0]
                cur.execute(
                    """
                    INSERT INTO sop_chunks
                        (document_id, document_version, chunk_index, heading_path,
                         content, category, sub_category, people, roles,
                         fine_grained_topics, embedding)
                    VALUES (%s, '1.0', 0, 'Approval', 'Vikas Sir approves WFH.',
                            'HR', 'WFH', %s, %s, %s, %s)
                    """,
                    (doc_id, ["Vikas Sir"], ["Team Lead"], ["approval"], [0.1] * 768),
                )
        conn.close()
        mock_embed_query.return_value = [0.1] * 768

        results = hybrid_search(
            "who approves WFH", limit=10, _filter_document_title=TEST_TITLE
        )
        match = results[0]

        assert "Vikas Sir" in match.people
        assert "Team Lead" in match.roles
        assert "approval" in match.fine_grained_topics
    finally:
        _cleanup()

import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.database.postgres import get_connection
from app.repositories.document_repository import upsert_document
from app.repositories.document_versions import get_all_document_versions

TITLES = ["TEST_VER_published", "TEST_VER_draft", "TEST_VER_inactive"]


def _cleanup():
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM sop_documents WHERE title = ANY(%s)", (TITLES,))
    finally:
        conn.close()


def _doc(title, version="1.0", status="Published", active="Active"):
    upsert_document(
        {"Title": title, "Category": "HR", "Sub Category": "WFH", "Version": version,
         "Status": status, "Is Active": active},
        [],
    )


def test_only_published_and_active_documents_are_returned():
    _cleanup()
    try:
        _doc("TEST_VER_published")
        _doc("TEST_VER_draft", status="Draft")
        _doc("TEST_VER_inactive", active="Inactive")

        versions = get_all_document_versions()

        assert versions["TEST_VER_published"] == "1.0"
        assert "TEST_VER_draft" not in versions
        assert "TEST_VER_inactive" not in versions
    finally:
        _cleanup()


def test_version_bump_is_reflected():
    """The mapping is part of every cache key, so a version bump here
    is what invalidates cached answers."""
    _cleanup()
    try:
        _doc("TEST_VER_published", version="1.0")
        before = get_all_document_versions()
        _doc("TEST_VER_published", version="2.0")
        after = get_all_document_versions()

        assert before["TEST_VER_published"] == "1.0"
        assert after["TEST_VER_published"] == "2.0"
        assert before != after
    finally:
        _cleanup()

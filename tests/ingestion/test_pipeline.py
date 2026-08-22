"""
Tests the pipeline orchestrator's logic (skip unchanged / insert new /
update changed) using a REAL local Postgres but MOCKED Gemini calls -
this catches wiring bugs between the pieces without needing a real API
key. The fully real version (real Gemini + real Postgres) is
scripts/run_ingestion.py, run manually by whoever has API credentials.
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.database.postgres import get_connection
from app.ingestion.pipeline import ingest_directory
from app.repositories.document_repository import get_document_version

SAMPLE_TXT = """Title: TEST_PIPELINE_DOC
Category: HR
Sub Category: WFH
Version: 1.0
Status: Published
Is Active: Active

Purpose
This is a test document for pipeline integration testing.

Approval Process
Requests must be approved by the Team Lead.
"""

SAMPLE_TXT_V2 = SAMPLE_TXT.replace("Version: 1.0", "Version: 2.0").replace(
    "Requests must be approved by the Team Lead.",
    "Requests must be approved by the Team Lead and Vikas Sir.",
)


def _cleanup():
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM sop_documents WHERE title = 'TEST_PIPELINE_DOC'")
    finally:
        conn.close()


def _mock_gemini_calls():
    """Patches the actual Gemini-calling functions (not the client
    itself) so the pipeline's own logic runs for real, only the
    external API calls are faked."""
    from app.ingestion.processors.entity_tagger import EntityTags

    tag_patch = patch(
        "app.ingestion.pipeline.tag_entities_for_chunks",
        return_value={
            0: EntityTags(people=[], roles=[], fine_grained_topics=[], category="HR", sub_category="WFH"),
            1: EntityTags(people=["Vikas Sir"], roles=["Team Lead"], fine_grained_topics=["approval"], category="HR", sub_category="WFH"),
        },
    )
    embed_patch = patch(
        "app.ingestion.pipeline.embed_chunks",
        return_value={0: [0.1] * 768, 1: [0.2] * 768},
    )
    return tag_patch, embed_patch


def test_first_ingest_inserts_new_document():
    _cleanup()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.txt"), "w") as f:
                f.write(SAMPLE_TXT)

            tag_patch, embed_patch = _mock_gemini_calls()
            with tag_patch, embed_patch:
                summary = ingest_directory(tmpdir, verbose=False)

            assert summary["inserted"] == 1
            assert summary["updated"] == 0
            assert summary["errors"] == []
    finally:
        _cleanup()


def test_second_ingest_of_same_version_is_skipped():
    """Proves the version-check actually prevents wasted re-processing
    (and re-paying for API calls) when nothing changed."""
    _cleanup()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.txt"), "w") as f:
                f.write(SAMPLE_TXT)

            tag_patch, embed_patch = _mock_gemini_calls()
            with tag_patch, embed_patch:
                ingest_directory(tmpdir, verbose=False)
                # run again, same version - should skip, and critically
                # should NOT call the (mocked) AI functions again
                with patch("app.ingestion.pipeline.tag_entities_for_chunks") as mock_tag:
                    summary = ingest_directory(tmpdir, verbose=False)
                    mock_tag.assert_not_called()

            assert summary["skipped_unchanged"] == 1
            assert summary["inserted"] == 0
    finally:
        _cleanup()


def test_flagged_structure_file_is_skipped_not_ingested():
    """The real point of the structure validator: a badly-flattened
    file (using the actual raw STLC excerpt structure) should be
    flagged and skipped, never silently chunked and stored."""
    _cleanup()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Same structural defect as the real raw STLC text: a
            # heading ("Entry Criteria") glued mid-paragraph with no
            # blank line before it, and no bullets to compensate.
            messy_txt = """Title: TEST_PIPELINE_DOC
Category: HR
Sub Category: WFH
Version: 1.0
Status: Published
Is Active: Active

Purpose
This is a test document. Entry Criteria Before testing begins many things must be checked and confirmed by the team including access and environment readiness across the board for every single release cycle without exception."""
            with open(os.path.join(tmpdir, "messy.txt"), "w") as f:
                f.write(messy_txt)

            tag_patch, embed_patch = _mock_gemini_calls()
            with tag_patch, embed_patch as mock_embed:
                summary = ingest_directory(tmpdir, verbose=False)

            assert summary["flagged_structure"] == 1
            assert summary["inserted"] == 0
            assert get_document_version("TEST_PIPELINE_DOC") is None
    finally:
        _cleanup()
    """The real end-to-end proof: old version's data is gone, new
    version's data (with the changed fact) is what's actually stored."""
    _cleanup()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")

            with open(path, "w") as f:
                f.write(SAMPLE_TXT)
            tag_patch, embed_patch = _mock_gemini_calls()
            with tag_patch, embed_patch:
                ingest_directory(tmpdir, verbose=False)

            with open(path, "w") as f:
                f.write(SAMPLE_TXT_V2)
            with tag_patch, embed_patch:
                summary = ingest_directory(tmpdir, verbose=False)

            assert summary["updated"] == 1

            conn = get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT version FROM sop_documents WHERE title = 'TEST_PIPELINE_DOC'"
                )
                assert cur.fetchone()[0] == "2.0"

                cur.execute(
                    "SELECT content FROM sop_chunks c "
                    "JOIN sop_documents d ON d.id = c.document_id "
                    "WHERE d.title = 'TEST_PIPELINE_DOC' AND c.heading_path LIKE '%Approval%'"
                )
                content = cur.fetchone()[0]
                assert "Vikas Sir" in content  # the NEW fact
            conn.close()
    finally:
        _cleanup()

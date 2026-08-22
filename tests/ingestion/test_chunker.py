import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.ingestion.loaders.txt_loader import TxtLoader
from app.ingestion.processors.chunker import chunk_sop

SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "samples"))


def _chunks_for(filename):
    loader = TxtLoader()
    doc = loader.load(os.path.join(SAMPLES_DIR, filename))
    return chunk_sop(doc.body, doc.metadata.get("Title", ""))


def test_no_chunk_is_empty():
    for filename in ["testing_process.txt", "wfh_policy.txt", "sdlc_module.txt"]:
        for c in _chunks_for(filename):
            assert c.content.strip() != ""


def test_bullet_list_never_separated_from_its_heading():
    """Regression test for the core rule-unit requirement: a heading's
    lead-in sentence and its bullets must be in the SAME chunk."""
    chunks = _chunks_for("testing_process.txt")
    entry_criteria = next(c for c in chunks if "Entry Criteria" in c.heading_path)
    assert "Requirement is documented" in entry_criteria.content
    assert "Test environment is available" in entry_criteria.content


def test_numbered_faq_items_keep_their_group_label():
    """Regression test for the fix where items 2-9 were falling back to
    the document title instead of keeping 'Intern WFH Policy - FAQ' /
    'General WFH Policy - FAQ' as their heading_path."""
    chunks = _chunks_for("wfh_policy.txt")
    assert len(chunks) == 9

    intern_items = [c for c in chunks if "Intern WFH Policy" in c.heading_path]
    general_items = [c for c in chunks if "General WFH Policy" in c.heading_path]
    assert len(intern_items) == 4
    assert len(general_items) == 5


def test_question_never_separated_from_its_answer():
    chunks = _chunks_for("wfh_policy.txt")
    q7 = next(c for c in chunks if "authority to approve WFH" in c.content)
    assert "Vikas Sir" in q7.content


def test_ambiguous_nesting_fails_safely_not_wrong():
    """Regression test for the Retesting/Regression/Test Closure case.
    We deliberately do NOT try to guess 3-level nesting from plain text.
    The important invariant: content boundaries stay correct even when
    the heading LABEL is imperfect. Test Closure's own content must
    never bleed into the Regression Testing chunk or vice versa."""
    chunks = _chunks_for("testing_process.txt")
    regression = next(c for c in chunks if c.content.startswith("Regression Testing"))
    closure = next(c for c in chunks if c.content.startswith("Test Closure"))
    assert "QA Sign-off" in closure.content
    assert "QA Sign-off" not in regression.content
    assert "OneDrive folder" in regression.content
    assert "OneDrive folder" not in closure.content or closure is regression


def test_sdlc_flat_structure_all_top_level():
    """SDLC sample has no sub-headings at all - every chunk should be a
    simple top-level entry, the easy case, should never mis-nest."""
    chunks = _chunks_for("sdlc_module.txt")
    assert len(chunks) == 17
    for c in chunks:
        assert ">" not in c.heading_path


def test_total_chunk_counts_stable():
    """Locks in the known-good chunk counts as of this build. If a future
    change to the chunker alters these counts, it should be a deliberate,
    reviewed change - not a silent regression."""
    assert len(_chunks_for("testing_process.txt")) == 12
    assert len(_chunks_for("wfh_policy.txt")) == 9
    assert len(_chunks_for("sdlc_module.txt")) == 17

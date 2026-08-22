import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.ingestion.loaders.txt_loader import TxtLoader
from app.ingestion.processors.structure_validator import assess_structure_quality

SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "samples"))

# This is the actual raw text Sir originally shared for the Testing
# Process / STLC SOP, BEFORE it was hand-cleaned into samples/testing_process.txt.
# Real evidence of what a flattened export looks like - almost no blank
# lines, headings and lists all run together in continuous prose.
RAW_FLATTENED_STLC_EXCERPT = """Title: Testing Process
Purpose
The purpose of STLC is to ensure that software is tested systematically, defects are identified and resolved, and the delivered product meets business and user requirements.
Entry Criteria
Before testing begins, ensure:

* Requirement is documented and available in Meld-EP / Task.
* Requirement understanding is completed.
* Necessary access is provided.
* Test environment is available.
Test Planning
Test Plan Preparation
QA team shall prepare a Test Plan containing:

* Scope of testing
* Testing approach
* Resources involved
* Timeline
* OneDrive folder path for all testing-related documents
Document Repository
All testing-related documents must be maintained in a centralized OneDrive folder, including:

* Test Plan
* Test Cases
* Test Execution Reports
* Defect Reports
* Release-wise Regression Testing Reports
The OneDrive folder path should be mentioned in the Test Plan document.
Test Case Design
Before writing test cases, QA should understand:

* Business process driving the requirement
* End user and user roles
* Use cases and workflows
* Access permissions
* UI look and feel requirements
* Functional requirements
* Validations
* Data requirements"""


def test_raw_flattened_stlc_is_flagged_unreliable():
    """The real, concrete negative case: the actual raw text Sir shared
    should be flagged, since it's genuinely what caused the original
    "splitting will be proper, not the way we currently doing" feedback."""
    report = assess_structure_quality(RAW_FLATTENED_STLC_EXCERPT)
    assert report.is_reliable is False
    assert len(report.warnings) > 0


def test_hand_cleaned_testing_process_is_reliable():
    """The same document, after we hand-cleaned it, should NOT be
    flagged - proving the validator distinguishes real structural
    quality, not just document length or topic."""
    loader = TxtLoader()
    doc = loader.load(os.path.join(SAMPLES_DIR, "testing_process.txt"))
    report = assess_structure_quality(doc.body)
    assert report.is_reliable is True
    assert report.warnings == []


def test_hand_cleaned_wfh_policy_is_reliable():
    loader = TxtLoader()
    doc = loader.load(os.path.join(SAMPLES_DIR, "wfh_policy.txt"))
    report = assess_structure_quality(doc.body)
    assert report.is_reliable is True


def test_hand_cleaned_sdlc_module_is_reliable():
    loader = TxtLoader()
    doc = loader.load(os.path.join(SAMPLES_DIR, "sdlc_module.txt"))
    report = assess_structure_quality(doc.body)
    assert report.is_reliable is True


def test_report_includes_actionable_detail_not_just_pass_fail():
    """Warnings should explain WHY, not just flag true/false - this is
    what makes the report useful for a human reviewing a flagged file."""
    report = assess_structure_quality(RAW_FLATTENED_STLC_EXCERPT)
    assert any("heading-shaped line" in w.lower() for w in report.warnings)


def test_mid_block_heading_detection_is_the_real_signal():
    """This is the actual, concrete defect found in the real raw text:
    'Document Repository' and 'Test Case Design' appear as headings
    glued mid-paragraph, not blank-line-separated. This is a more
    precise, direct signal than blank-line ratio or block length alone
    - both of those under-triggered on this exact real example during
    development, which is why this check was added."""
    report = assess_structure_quality(RAW_FLATTENED_STLC_EXCERPT)
    assert report.mid_block_heading_count > 0


def test_empty_body_does_not_crash():
    report = assess_structure_quality("")
    assert isinstance(report.is_reliable, bool)


def test_short_body_with_no_bullets_is_not_falsely_flagged():
    """A short, simple document with no lists at all shouldn't be
    flagged just for being short - the bullet-check only applies to
    substantial documents (over 100 words), see the >100 word threshold
    in the validator itself."""
    short_body = "Purpose\nThis policy applies to all employees in the organization."
    report = assess_structure_quality(short_body)
    # Should not warn about missing bullets for such a short document
    assert not any("bullet" in w.lower() for w in report.warnings)

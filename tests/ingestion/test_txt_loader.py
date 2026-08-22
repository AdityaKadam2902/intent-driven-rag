import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.ingestion.loaders.txt_loader import TxtLoader

SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "samples"))


def test_loads_all_three_samples():
    loader = TxtLoader()
    docs = loader.load_all(SAMPLES_DIR)
    assert len(docs) == 3


def test_metadata_parsed_for_testing_process():
    loader = TxtLoader()
    doc = loader.load(os.path.join(SAMPLES_DIR, "testing_process.txt"))
    assert doc.metadata["Title"] == "Testing Process"
    assert doc.metadata["Category"] == "Engineer"
    assert doc.metadata["Sub Category"] == "SDLC"
    assert doc.metadata["Version"] == "1.0"
    assert doc.metadata["Status"] == "Published"
    assert doc.metadata["Is Active"] == "Active"


def test_metadata_field_count_differs_per_file_generic_not_hardcoded():
    """WFH and SDLC samples have fewer metadata fields (no Created/Updated
    By/Date) than testing_process - the parser must not assume a fixed
    field count or fixed set of keys."""
    loader = TxtLoader()
    wfh = loader.load(os.path.join(SAMPLES_DIR, "wfh_policy.txt"))
    stlc = loader.load(os.path.join(SAMPLES_DIR, "testing_process.txt"))
    assert len(wfh.metadata) < len(stlc.metadata)
    assert wfh.metadata["Title"] == "Work From Home (WFH) Policy"


def test_body_excludes_metadata_block():
    loader = TxtLoader()
    doc = loader.load(os.path.join(SAMPLES_DIR, "testing_process.txt"))
    assert "Version:" not in doc.body
    assert doc.body.startswith("Purpose")


def test_body_is_never_empty_for_valid_files():
    loader = TxtLoader()
    docs = loader.load_all(SAMPLES_DIR)
    for doc in docs:
        assert len(doc.body) > 0

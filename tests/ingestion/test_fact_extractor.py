"""
These tests mock the Gemini client so they run without a real API key -
they check that our code calls the SDK correctly and handles the
response correctly, not that Gemini itself extracts facts well (that
needs a live call - see scripts/test_fact_extraction_live.py for that,
run manually with a real GEMINI_API_KEY).
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.ingestion.processors.fact_extractor import (
    ExtractedFact,
    FactExtractionResult,
    extract_facts,
)


def _fake_response(facts: list[ExtractedFact]):
    fake = MagicMock()
    fake.parsed = FactExtractionResult(facts=facts)
    return fake


@patch("app.ingestion.processors.fact_extractor.get_client")
def test_extract_facts_returns_parsed_list(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response(
        [
            ExtractedFact(
                policy="wfh_approver",
                value="Team Lead and Vikas Sir",
                condition="reviewed by Team Lead, approved by Vikas Sir",
                source_heading="Who has the authority to approve WFH requests?",
            )
        ]
    )
    mock_get_client.return_value = mock_client

    facts = extract_facts("some WFH content", doc_title="WFH Policy")

    assert len(facts) == 1
    assert facts[0].policy == "wfh_approver"
    assert "Vikas Sir" in facts[0].value


@patch("app.ingestion.processors.fact_extractor.get_client")
def test_extract_facts_handles_zero_facts(mock_get_client):
    """Purely descriptive content with no rules/numbers should return an
    empty list, not an error - this is a valid, expected outcome."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response([])
    mock_get_client.return_value = mock_client

    facts = extract_facts("just a general description with no rules", doc_title="X")

    assert facts == []


@patch("app.ingestion.processors.fact_extractor.get_client")
def test_extract_facts_calls_with_deterministic_temperature(mock_get_client):
    """Extraction must use temperature=0.0 - this is a precision task,
    not creative writing, and should be as deterministic as possible."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response([])
    mock_get_client.return_value = mock_client

    extract_facts("content", doc_title="X")

    _, kwargs = mock_client.models.generate_content.call_args
    assert kwargs["config"].temperature == 0.0


@patch("app.ingestion.processors.fact_extractor.get_client")
def test_extract_facts_uses_json_response_schema(mock_get_client):
    """Must request structured JSON output, not free-form text - this
    is what makes the output reliably parseable."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response([])
    mock_get_client.return_value = mock_client

    extract_facts("content", doc_title="X")

    _, kwargs = mock_client.models.generate_content.call_args
    assert kwargs["config"].response_mime_type == "application/json"
    assert kwargs["config"].response_schema == FactExtractionResult

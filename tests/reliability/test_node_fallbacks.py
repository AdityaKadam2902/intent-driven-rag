"""
Proves the actual failure mode from the real bug report is fixed: when
Gemini is unavailable, nodes must return a graceful fallback state
update, never let a raw exception surface (which is what produced the
"Something went wrong answering that question" error in the live web
app before this circuit breaker existed).
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.graph.nodes.generation.synthesize import synthesize
from app.graph.nodes.retrieval.resolve_follow_up import resolve_follow_up
from app.graph.nodes.routing.classify_intention import classify_intention
from app.llm.gemini import reset_breaker
from app.reliability.circuit_breaker import CircuitOpenError


def _make_fake_chunk():
    from dataclasses import dataclass

    @dataclass
    class FakeChunk:
        document_title: str = "WFH Policy"
        document_version: str = "1.0"
        heading_path: str = "Approval"
        content: str = "Team Lead and Vikas Sir must approve."
        combined_score: float = 0.7

    return FakeChunk()


def setup_function():
    reset_breaker()


@patch("app.graph.nodes.routing.classify_intention.generate_content_safe")
def test_classify_intention_falls_back_to_informational_when_breaker_open(mock_gen):
    mock_gen.side_effect = CircuitOpenError("breaker open")

    result = classify_intention({"user_message": "what is the WFH policy"})

    assert result == {"intention": "informational"}


@patch("app.graph.nodes.retrieval.resolve_follow_up.generate_content_safe")
def test_resolve_follow_up_falls_back_to_raw_message_when_breaker_open(mock_gen):
    mock_gen.side_effect = CircuitOpenError("breaker open")

    result = resolve_follow_up({
        "user_message": "what about for interns",
        "conversation_history": [{"role": "assistant", "content": "WFH needs approval."}],
    })

    assert result == {"resolved_query": "what about for interns"}


@patch("app.graph.nodes.generation.synthesize.generate_content_safe")
def test_synthesize_falls_back_to_source_list_when_breaker_open(mock_gen):
    """This is THE regression test for the real bug: no raw error
    should ever reach the user - a graceful, honest fallback instead."""
    mock_gen.side_effect = CircuitOpenError("breaker open")

    chunks = [_make_fake_chunk()]
    result = synthesize({"user_message": "who approves WFH", "retrieved_chunks": chunks})

    assert "trouble generating" in result["answer"]
    assert "WFH Policy" in result["answer"]
    assert result["confidence"] == "Low"
    assert len(result["sources"]) == 1


@patch("app.graph.nodes.generation.synthesize.generate_content_safe")
def test_synthesize_never_raises_even_when_breaker_open(mock_gen):
    """The node must catch CircuitOpenError itself - it must never
    propagate up to the web layer as an unhandled exception."""
    mock_gen.side_effect = CircuitOpenError("breaker open")

    chunks = [_make_fake_chunk()]
    try:
        result = synthesize({"user_message": "test", "retrieved_chunks": chunks})
    except CircuitOpenError:
        assert False, "synthesize() must not let CircuitOpenError propagate"

    assert result is not None

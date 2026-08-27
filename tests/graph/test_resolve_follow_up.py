import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.graph.nodes.retrieval.resolve_follow_up import ResolvedQuery, resolve_follow_up


def _fake_response(resolved_query: str):
    fake = MagicMock()
    fake.parsed = ResolvedQuery(resolved_query=resolved_query)
    return fake


@patch("app.graph.nodes.retrieval.resolve_follow_up.generate_content_safe")
def test_resolves_using_previous_message(mock_generate):
    mock_generate.return_value = _fake_response("What is the WFH policy for interns?")

    result = resolve_follow_up({
        "user_message": "what about for interns",
        "conversation_history": [{"role": "assistant", "content": "The WFH policy requires approval."}],
    })

    assert result == {"resolved_query": "What is the WFH policy for interns?"}


def test_falls_back_to_raw_message_when_no_history():
    """No history to resolve against - must not crash, must not call
    the LLM needlessly, and should just pass the raw message through."""
    with patch("app.graph.nodes.retrieval.resolve_follow_up.generate_content_safe") as mock_generate:
        result = resolve_follow_up({"user_message": "what about for interns"})

        assert result == {"resolved_query": "what about for interns"}
        mock_generate.assert_not_called()


@patch("app.graph.nodes.retrieval.resolve_follow_up.generate_content_safe")
def test_includes_previous_message_in_prompt(mock_generate):
    mock_generate.return_value = _fake_response("resolved")

    resolve_follow_up({
        "user_message": "what about for interns",
        "conversation_history": [{"role": "assistant", "content": "UNIQUE_MARKER_TEXT"}],
    })

    _, kwargs = mock_generate.call_args
    assert "UNIQUE_MARKER_TEXT" in kwargs["contents"]


@patch("app.graph.nodes.retrieval.resolve_follow_up.generate_content_safe")
def test_uses_deterministic_temperature(mock_generate):
    mock_generate.return_value = _fake_response("resolved")

    resolve_follow_up({
        "user_message": "what about for interns",
        "conversation_history": [{"role": "assistant", "content": "context"}],
    })

    _, kwargs = mock_generate.call_args
    assert kwargs["config"].temperature == 0.0

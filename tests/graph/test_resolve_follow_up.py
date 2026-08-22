import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.graph.nodes.retrieval.resolve_follow_up import ResolvedQuery, resolve_follow_up


def _fake_response(resolved_query: str):
    fake = MagicMock()
    fake.parsed = ResolvedQuery(resolved_query=resolved_query)
    return fake


@patch("app.graph.nodes.retrieval.resolve_follow_up.get_client")
def test_resolves_using_previous_message(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response(
        "What is the WFH policy for interns?"
    )
    mock_get_client.return_value = mock_client

    result = resolve_follow_up({
        "user_message": "what about for interns",
        "conversation_history": [{"role": "assistant", "content": "The WFH policy requires approval."}],
    })

    assert result == {"resolved_query": "What is the WFH policy for interns?"}


@patch("app.graph.nodes.retrieval.resolve_follow_up.get_client")
def test_falls_back_to_raw_message_when_no_history(mock_get_client):
    """No history to resolve against - must not crash, must not call
    the LLM needlessly, and should just pass the raw message through."""
    result = resolve_follow_up({"user_message": "what about for interns"})

    assert result == {"resolved_query": "what about for interns"}
    mock_get_client.assert_not_called()


@patch("app.graph.nodes.retrieval.resolve_follow_up.get_client")
def test_includes_previous_message_in_prompt(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response("resolved")
    mock_get_client.return_value = mock_client

    resolve_follow_up({
        "user_message": "what about for interns",
        "conversation_history": [{"role": "assistant", "content": "UNIQUE_MARKER_TEXT"}],
    })

    args, kwargs = mock_client.models.generate_content.call_args
    assert "UNIQUE_MARKER_TEXT" in kwargs["contents"]


@patch("app.graph.nodes.retrieval.resolve_follow_up.get_client")
def test_uses_deterministic_temperature(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response("resolved")
    mock_get_client.return_value = mock_client

    resolve_follow_up({
        "user_message": "what about for interns",
        "conversation_history": [{"role": "assistant", "content": "context"}],
    })

    _, kwargs = mock_client.models.generate_content.call_args
    assert kwargs["config"].temperature == 0.0

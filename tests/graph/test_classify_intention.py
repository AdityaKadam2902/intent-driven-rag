import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.graph.nodes.routing.classify_intention import (
    IntentionClassification,
    classify_intention,
)


def _fake_response(intention: str):
    fake = MagicMock()
    fake.parsed = IntentionClassification(intention=intention)
    return fake


@patch("app.graph.nodes.routing.classify_intention.get_client")
def test_returns_partial_state_update_not_full_state(mock_get_client):
    """LangGraph convention: a node returns a partial dict to be merged,
    not the whole state object."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response("chitchat")
    mock_get_client.return_value = mock_client

    result = classify_intention({"user_message": "hi"})

    assert result == {"intention": "chitchat"}


@patch("app.graph.nodes.routing.classify_intention.get_client")
def test_uses_deterministic_temperature(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response("informational")
    mock_get_client.return_value = mock_client

    classify_intention({"user_message": "what is the WFH policy"})

    _, kwargs = mock_client.models.generate_content.call_args
    assert kwargs["config"].temperature == 0.0


@patch("app.graph.nodes.routing.classify_intention.get_client")
def test_includes_previous_message_when_history_exists(mock_get_client):
    """Follow-up detection needs the previous turn - confirm it's
    actually included in the prompt when history is present."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response("follow_up")
    mock_get_client.return_value = mock_client

    classify_intention({
        "user_message": "what about for interns",
        "conversation_history": [{"role": "assistant", "content": "The WFH policy requires Team Lead approval."}],
    })

    args, kwargs = mock_client.models.generate_content.call_args
    prompt_sent = kwargs["contents"]
    assert "Team Lead approval" in prompt_sent


@patch("app.graph.nodes.routing.classify_intention.get_client")
def test_handles_no_history_gracefully(mock_get_client):
    """First message in a conversation has no history - must not crash."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response("chitchat")
    mock_get_client.return_value = mock_client

    result = classify_intention({"user_message": "hi"})

    assert result["intention"] == "chitchat"


@patch("app.graph.nodes.routing.classify_intention.get_client")
def test_uses_json_response_schema(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response("meta")
    mock_get_client.return_value = mock_client

    classify_intention({"user_message": "what can you help with"})

    _, kwargs = mock_client.models.generate_content.call_args
    assert kwargs["config"].response_mime_type == "application/json"
    assert kwargs["config"].response_schema == IntentionClassification

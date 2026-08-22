import pytest

import web.app.chat.routes as chat_routes


class _FakeHistoryStore:
    """In-memory stand-in for the chat_turns table, keyed like the real one."""

    def __init__(self):
        self.turns = {}  # (session_id, assistant_id) -> list[dict]

    def _key(self, session_id, assistant_id):
        return (session_id, assistant_id)

    def get_conversation_history(self, session_id, assistant_id):
        rows = self.turns.get(self._key(session_id, assistant_id), [])
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def get_display_turns(self, session_id, assistant_id):
        return list(self.turns.get(self._key(session_id, assistant_id), []))

    def append_user_turn(self, session_id, assistant_id, content):
        key = self._key(session_id, assistant_id)
        self.turns.setdefault(key, []).append({"role": "user", "content": content})

    def append_assistant_turn(
        self, session_id, assistant_id, content, confidence, sources, intention
    ):
        key = self._key(session_id, assistant_id)
        self.turns.setdefault(key, []).append(
            {
                "role": "assistant",
                "content": content,
                "confidence": confidence,
                "sources": sources,
                "intention": intention,
            }
        )


    def clear_history(self, session_id, assistant_id):
        self.turns.pop(self._key(session_id, assistant_id), None)


@pytest.fixture
def fake_store(monkeypatch):
    store = _FakeHistoryStore()
    monkeypatch.setattr(chat_routes, "get_conversation_history", store.get_conversation_history)
    monkeypatch.setattr(chat_routes, "get_display_turns", store.get_display_turns)
    monkeypatch.setattr(chat_routes, "append_user_turn", store.append_user_turn)
    monkeypatch.setattr(chat_routes, "append_assistant_turn", store.append_assistant_turn)
    monkeypatch.setattr(chat_routes, "clear_history", store.clear_history)
    return store


def test_chat_page_requires_login(client):
    resp = client.get("/chat/sop_assistant", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login"


def test_chat_page_unknown_assistant_redirects_home(logged_in_client, fake_store):
    resp = logged_in_client.get("/chat/does_not_exist", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/"


def test_chat_page_renders_empty_state(logged_in_client, fake_store):
    resp = logged_in_client.get("/chat/sop_assistant")
    assert resp.status_code == 200
    assert "No messages yet" in resp.text


def test_ask_question_appends_turns_and_renders_answer(logged_in_client, fake_store):
    resp = logged_in_client.post(
        "/chat/sop_assistant", data={"message": "What is the onboarding process?"}
    )
    assert resp.status_code == 200
    assert "What is the onboarding process?" in resp.text
    assert resp.status_code == 200
    assert "Confidence:" in resp.text
    assert "Sources:" in resp.text


def test_conversation_history_passed_to_second_call(logged_in_client, fake_store):
    logged_in_client.post("/chat/sop_assistant", data={"message": "first question"})
    logged_in_client.post("/chat/sop_assistant", data={"message": "second question"})

    key = list(fake_store.turns.keys())[0]
    roles = [t["role"] for t in fake_store.turns[key]]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_clear_chat_removes_history(logged_in_client, fake_store):
    logged_in_client.post("/chat/sop_assistant", data={"message": "a question"})
    key = list(fake_store.turns.keys())[0]
    assert len(fake_store.turns[key]) == 2

    resp = logged_in_client.post("/chat/sop_assistant/clear", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/chat/sop_assistant"
    assert fake_store.turns.get(key, []) == []

    resp = logged_in_client.get("/chat/sop_assistant")
    assert "No messages yet" in resp.text


def test_graph_failure_shows_error_without_crashing(logged_in_client, fake_store, monkeypatch):
    from web.app.assistants import registry

    def boom():
        class _Boom:
            def invoke(self, payload):
                raise RuntimeError("Gemini API unavailable")
        return _Boom()

    monkeypatch.setitem(
        registry.ASSISTANT_REGISTRY,
        "sop_assistant",
        registry.AssistantConfig(
            id="sop_assistant",
            name="SOP Assistant",
            description="test",
            build_graph=boom,
            ingest_directory=lambda d: None,
            list_documents=lambda: [],
        ),
    )

    resp = logged_in_client.post("/chat/sop_assistant", data={"message": "trigger failure"})
    assert resp.status_code == 200
    assert "Something went wrong" in resp.text

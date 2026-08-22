"""
Owns the chat_turns table. Two read shapes are exposed deliberately:

- get_conversation_history(): plain {"role", "content"} dicts, the shape
  build_graph() expects as conversation_history.
- get_display_turns(): full rows (confidence, sources, intention included)
  for rendering the chat template.

Keeping these separate means a change to what the template shows never
risks leaking extra fields into the graph invocation payload.
"""
import json

from web.app.db import get_connection


def get_conversation_history(session_id: str, assistant_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM chat_turns
            WHERE session_id = %s AND assistant_id = %s
            ORDER BY turn_index ASC
            """,
            (session_id, assistant_id),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def get_display_turns(session_id: str, assistant_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content, confidence, sources, intention, created_at
            FROM chat_turns
            WHERE session_id = %s AND assistant_id = %s
            ORDER BY turn_index ASC
            """,
            (session_id, assistant_id),
        ).fetchall()
    return list(rows)


def _next_turn_index(conn, session_id: str, assistant_id: str) -> int:
    row = conn.execute(
        """
        SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_index
        FROM chat_turns
        WHERE session_id = %s AND assistant_id = %s
        """,
        (session_id, assistant_id),
    ).fetchone()
    return row["next_index"]


def append_user_turn(session_id: str, assistant_id: str, content: str) -> None:
    with get_connection() as conn:
        idx = _next_turn_index(conn, session_id, assistant_id)
        conn.execute(
            """
            INSERT INTO chat_turns (session_id, assistant_id, turn_index, role, content)
            VALUES (%s, %s, %s, 'user', %s)
            """,
            (session_id, assistant_id, idx, content),
        )


def append_assistant_turn(
    session_id: str,
    assistant_id: str,
    content: str,
    confidence: str | None,
    sources: list | None,
    intention: str | None,
) -> None:
    with get_connection() as conn:
        idx = _next_turn_index(conn, session_id, assistant_id)
        conn.execute(
            """
            INSERT INTO chat_turns
                (session_id, assistant_id, turn_index, role, content, confidence, sources, intention)
            VALUES (%s, %s, %s, 'assistant', %s, %s, %s, %s)
            """,
            (
                session_id,
                assistant_id,
                idx,
                content,
                confidence,
                json.dumps(sources) if sources is not None else None,
                intention,
            ),
        )


def clear_history(session_id: str, assistant_id: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            DELETE FROM chat_turns
            WHERE session_id = %s AND assistant_id = %s
            """,
            (session_id, assistant_id),
        )
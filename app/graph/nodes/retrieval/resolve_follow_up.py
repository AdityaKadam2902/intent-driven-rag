"""
Follow-up resolution - Section 3's "Load history, then retrieve" path.

Circuit breaker: if Gemini is down, falls back to using the raw
follow-up message as the search query directly (worse recall than a
properly resolved query, but never a crash).
"""
from pydantic import BaseModel, Field
from google.genai import types

from app.graph.state import GraphState
from app.llm.gemini import DEFAULT_MODEL, generate_content_safe
from app.reliability.circuit_breaker import CircuitOpenError


class ResolvedQuery(BaseModel):
    resolved_query: str = Field(
        description="The follow-up question rewritten as a standalone "
        "question, incorporating the necessary context from the "
        "previous message so it can be searched on its own."
    )


RESOLVE_PROMPT = """The user sent a follow-up message that only makes sense in context of the previous turn. Rewrite it as a standalone question that includes the necessary context, so it can be searched without needing the conversation history.

Previous message: {previous_message}

Follow-up message: {message}

Rewrite the follow-up as a complete, standalone question.
"""


def resolve_follow_up(state: GraphState) -> dict:
    history = state.get("conversation_history", [])
    previous_message = history[-1]["content"] if history else ""

    if not previous_message:
        return {"resolved_query": state["user_message"]}

    try:
        response = generate_content_safe(
            model=DEFAULT_MODEL,
            contents=RESOLVE_PROMPT.format(
                previous_message=previous_message, message=state["user_message"]
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ResolvedQuery,
                temperature=0.0,
            ),
        )
    except CircuitOpenError:
        # Gemini is down - fall back to the raw follow-up text as the
        # search query. Worse recall (less context in the query) but
        # never a crash.
        return {"resolved_query": state["user_message"]}

    result: ResolvedQuery = response.parsed
    return {"resolved_query": result.resolved_query}

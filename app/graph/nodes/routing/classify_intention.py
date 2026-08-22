"""
Intention classification - Section 3 of the architecture doc.

A single LLM call that classifies the incoming message into exactly one
of six outcomes, BEFORE any retrieval happens.

Circuit breaker: if Gemini is down, classification falls back to
"informational" rather than crashing - a safe default, since it routes
into the full retrieval path, which itself has its own fallback (see
synthesize.py) if the breaker is still open by the time it gets there.
Never leave the user with a raw error for something as basic as
"what kind of message is this."
"""
from pydantic import BaseModel, Field
from google.genai import types

from app.graph.state import GraphState, IntentionLabel
from app.llm.gemini import DEFAULT_MODEL, generate_content_safe
from app.reliability.circuit_breaker import CircuitOpenError


class IntentionClassification(BaseModel):
    intention: IntentionLabel = Field(
        description="Exactly one of: chitchat, meta, informational, "
        "follow_up, actionable, escalation"
    )


CLASSIFICATION_PROMPT = """Classify this message into EXACTLY ONE of six categories.

Categories:
- chitchat: greetings, thanks, small talk with no real question (e.g. "Hi", "thanks!")
- meta: questions about the assistant itself, not about company SOPs (e.g. "what can you help with", "how confident are you")
- informational: a real question that needs looking up company SOP/policy content (e.g. "what's the WFH policy")
- follow_up: a question that only makes sense by referring to the PREVIOUS message in the conversation (e.g. "what about for interns" right after a question about leave)
- actionable: the user wants the assistant to actually DO something in Meldep, not just explain it (e.g. "apply leave for me", "submit this request")
- escalation: sensitive, urgent, or clearly needs a human (e.g. harassment, grievance, urgent personal crisis)

Previous message in the conversation (empty if this is the first message): {previous_message}

Message to classify: {message}
"""


def classify_intention(state: GraphState) -> dict:
    """LangGraph node. Calls the LLM once through the shared circuit
    breaker, and returns a state update with the classified intention."""
    history = state.get("conversation_history", [])
    previous_message = history[-1]["content"] if history else ""

    prompt = CLASSIFICATION_PROMPT.format(
        previous_message=previous_message,
        message=state["user_message"],
    )

    try:
        response = generate_content_safe(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=IntentionClassification,
                temperature=0.0,
            ),
        )
    except CircuitOpenError:
        # Gemini is down - default to "informational" rather than
        # crash. This routes into retrieve -> rerank -> synthesize,
        # which has its own breaker-aware fallback if Gemini is still
        # down by the time synthesis runs.
        return {"intention": "informational"}

    result: IntentionClassification = response.parsed
    return {"intention": result.intention}

"""
Shared state schema for the query-time LangGraph pipeline.
"""
from typing import Literal, TypedDict

IntentionLabel = Literal[
    "chitchat", "meta", "informational", "follow_up", "actionable", "escalation"
]


class GraphState(TypedDict, total=False):
    user_message: str
    conversation_history: list[dict]

    intention: IntentionLabel
    resolved_query: str
    retrieved_chunks: list
    db_unavailable: bool  # set by retrieve() when the Postgres circuit breaker is open

    answer: str
    confidence: Literal["High", "Medium", "Low"]
    sources: list[dict]

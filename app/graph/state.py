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
    db_unavailable: bool

    # Cache fields - set by cache_lookup, consumed by cache_write
    cache_hit: bool
    _cache_query_embedding: list
    _cache_query_entities: dict
    _cache_document_versions: dict

    answer: str
    confidence: Literal["High", "Medium", "Low"]
    sources: list[dict]

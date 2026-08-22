"""
Single source of truth for which assistants exist and how the web layer
reaches their RAG functions. Dashboard, chat routes, and documents routes
all read from this dict rather than hardcoding assistant-specific logic.

To add a new assistant later: add an entry here, then add its own
build_graph / ingest_directory implementation under app/. No other file
needs to change.
"""
from dataclasses import dataclass
from typing import Callable

from app.graph.builder import build_graph as sop_build_graph
from app.ingestion.pipeline import ingest_directory as sop_ingest_directory
from app.repositories.document_repository import (
    get_ingested_documents as sop_get_ingested_documents,
)


@dataclass(frozen=True)
class AssistantConfig:
    id: str
    name: str
    description: str
    build_graph: Callable
    ingest_directory: Callable
    list_documents: Callable


ASSISTANT_REGISTRY: dict[str, AssistantConfig] = {
    "sop_assistant": AssistantConfig(
        id="sop_assistant",
        name="SOP Assistant",
        description="Ask questions about Meldep Standard Operating Procedures.",
        build_graph=sop_build_graph,
        ingest_directory=sop_ingest_directory,
        list_documents=sop_get_ingested_documents,
    ),
}


def get_assistant(assistant_id: str) -> AssistantConfig:
    try:
        return ASSISTANT_REGISTRY[assistant_id]
    except KeyError:
        raise ValueError(f"Unknown assistant: {assistant_id}")

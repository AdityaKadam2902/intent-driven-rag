"""
Tests the COMPILED graph end to end - not just the classification node
in isolation, but that classify_intention's output actually routes
through the conditional edge to the correct terminal node. This is the
real proof that the six-way router is "provably real, not implied" -
Section 3's requirement - since these tests exercise the actual graph
execution, not just a function call.
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.graph.builder import build_graph
from app.graph.nodes.routing.classify_intention import IntentionClassification


def _fake_response(intention: str):
    fake = MagicMock()
    fake.parsed = IntentionClassification(intention=intention)
    return fake


@patch("app.graph.nodes.routing.classify_intention.get_client")
def test_chitchat_routes_to_direct_answer(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response("chitchat")
    mock_get_client.return_value = mock_client

    graph = build_graph()
    result = graph.invoke({"user_message": "hi"})

    assert result["intention"] == "chitchat"
    assert "SOPs" in result["answer"]
    assert result["sources"] == []


@patch("app.graph.nodes.routing.classify_intention.get_client")
def test_meta_routes_to_direct_answer(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response("meta")
    mock_get_client.return_value = mock_client

    graph = build_graph()
    result = graph.invoke({"user_message": "what can you help with"})

    assert result["intention"] == "meta"
    assert "cite" in result["answer"].lower()


@patch("app.graph.nodes.routing.classify_intention.get_client")
def test_actionable_routes_to_explain_only_response(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response("actionable")
    mock_get_client.return_value = mock_client

    graph = build_graph()
    result = graph.invoke({"user_message": "apply leave for me"})

    assert result["intention"] == "actionable"
    assert "can't perform actions" in result["answer"]


@patch("app.graph.nodes.routing.classify_intention.get_client")
def test_escalation_routes_to_human_handoff(mock_get_client):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response("escalation")
    mock_get_client.return_value = mock_client

    graph = build_graph()
    result = graph.invoke({"user_message": "I need to report harassment"})

    assert result["intention"] == "escalation"
    assert "HR" in result["answer"]


@patch("app.graph.nodes.generation.synthesize.get_client")
@patch("app.graph.nodes.retrieval.retrieve.hybrid_search")
@patch("app.graph.nodes.routing.classify_intention.get_client")
def test_informational_routes_through_full_retrieval_and_synthesis(
    mock_classify_client, mock_search, mock_synth_client
):
    from app.retrieval.retrievers.hybrid_retriever import RetrievedChunk

    mock_classify_client.return_value.models.generate_content.return_value = _fake_response("informational")

    chunk = RetrievedChunk(
        chunk_id="1", document_id="1", document_title="WFH Policy", document_version="1.0",
        heading_path="Approval", content="Vikas Sir and Team Lead approve WFH.",
        category="HR", sub_category="WFH", people=["Vikas Sir"], roles=["Team Lead"],
        fine_grained_topics=["approval"], vector_score=0.7, text_score=0.7, combined_score=0.7,
    )
    mock_search.return_value = [chunk]

    fake_llm_answer = MagicMock()
    fake_llm_answer.text = "Vikas Sir and the Team Lead approve WFH requests."
    mock_synth_client.return_value.models.generate_content.return_value = fake_llm_answer

    graph = build_graph()
    result = graph.invoke({"user_message": "who approves WFH"})

    assert result["intention"] == "informational"
    assert result["answer"] == "Vikas Sir and the Team Lead approve WFH requests."
    assert result["confidence"] == "High"
    assert result["sources"][0]["document_title"] == "WFH Policy"


@patch("app.graph.nodes.generation.synthesize.get_client")
@patch("app.graph.nodes.retrieval.retrieve.hybrid_search")
@patch("app.graph.nodes.retrieval.resolve_follow_up.get_client")
@patch("app.graph.nodes.routing.classify_intention.get_client")
def test_follow_up_resolves_then_retrieves_then_synthesizes(
    mock_classify_client, mock_resolve_client, mock_search, mock_synth_client
):
    """The full follow_up path: classify -> resolve_follow_up ->
    retrieve -> rerank -> synthesize, each step's mock proving the
    previous step's output was actually used by the next one."""
    from app.graph.nodes.retrieval.resolve_follow_up import ResolvedQuery
    from app.retrieval.retrievers.hybrid_retriever import RetrievedChunk

    mock_classify_client.return_value.models.generate_content.return_value = _fake_response("follow_up")

    fake_resolved = MagicMock()
    fake_resolved.parsed = ResolvedQuery(resolved_query="What is the WFH policy for interns?")
    mock_resolve_client.return_value.models.generate_content.return_value = fake_resolved

    chunk = RetrievedChunk(
        chunk_id="1", document_id="1", document_title="WFH Policy", document_version="1.0",
        heading_path="Intern WFH Policy - FAQ", content="Interns are not allowed WFH.",
        category="HR", sub_category="WFH", people=[], roles=["Intern"],
        fine_grained_topics=["eligibility"], vector_score=0.7, text_score=0.7, combined_score=0.7,
    )
    mock_search.return_value = [chunk]

    fake_llm_answer = MagicMock()
    fake_llm_answer.text = "Interns are generally not allowed to work from home."
    mock_synth_client.return_value.models.generate_content.return_value = fake_llm_answer

    graph = build_graph()
    result = graph.invoke({
        "user_message": "what about for interns",
        "conversation_history": [{"role": "assistant", "content": "WFH needs approval."}],
    })

    assert result["intention"] == "follow_up"
    assert result["resolved_query"] == "What is the WFH policy for interns?"
    # proves retrieve() actually used the resolved query, not the raw one
    args, _ = mock_search.call_args
    assert args[0] == "What is the WFH policy for interns?"
    assert result["answer"] == "Interns are generally not allowed to work from home."


@patch("app.graph.nodes.routing.classify_intention.get_client")
def test_chitchat_never_calls_retrieval_or_embedding(mock_get_client):
    """The actual architectural guarantee Section 3 requires: no-DB-call
    paths must never touch retrieval or embeddings at all, not just
    return a plausible-looking answer. Patches embed_query itself and
    asserts it is NEVER called for a chitchat message."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _fake_response("chitchat")
    mock_get_client.return_value = mock_client

    with patch("app.ingestion.embeddings.embedder.embed_query") as mock_embed:
        graph = build_graph()
        graph.invoke({"user_message": "thanks!"})
        mock_embed.assert_not_called()

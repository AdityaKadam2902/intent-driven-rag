"""
Proves: (1) the Postgres breaker is genuinely separate from the Gemini
breaker - tripping one must never affect the other, and (2) a DB
outage during retrieval produces the distinct, honest "database
connectivity issue" message, not the misleading "SOPs don't cover
this" message.
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.database.postgres import (
    get_db_breaker_state,
    reset_db_breaker,
    with_db_breaker,
)
from app.graph.nodes.generation.synthesize import synthesize
from app.graph.nodes.retrieval.retrieve import retrieve
from app.llm.gemini import get_breaker_state, reset_breaker
from app.reliability.circuit_breaker import CircuitOpenError


def setup_function():
    reset_breaker()
    reset_db_breaker()


def test_postgres_breaker_starts_closed():
    assert get_db_breaker_state() == "closed"


def test_postgres_breaker_opens_after_repeated_failures():
    def failing():
        raise RuntimeError("connection refused")

    for _ in range(5):
        try:
            with_db_breaker(failing)
        except RuntimeError:
            pass

    assert get_db_breaker_state() == "open"


def test_postgres_breaker_failure_does_not_affect_gemini_breaker():
    """The core isolation guarantee: tripping the DB breaker must
    never trip or otherwise affect the separate Gemini breaker."""
    def failing():
        raise RuntimeError("connection refused")

    for _ in range(5):
        try:
            with_db_breaker(failing)
        except RuntimeError:
            pass

    assert get_db_breaker_state() == "open"
    assert get_breaker_state() == "closed"  # Gemini breaker unaffected


@patch("app.graph.nodes.retrieval.retrieve.hybrid_search")
def test_retrieve_node_sets_db_unavailable_flag_on_circuit_open(mock_search):
    mock_search.side_effect = CircuitOpenError("db breaker open")

    result = retrieve({"user_message": "who approves WFH"})

    assert result["retrieved_chunks"] == []
    assert result["db_unavailable"] is True


@patch("app.graph.nodes.retrieval.retrieve.hybrid_search")
def test_retrieve_node_does_not_set_flag_on_success(mock_search):
    mock_search.return_value = []

    result = retrieve({"user_message": "test"})

    assert "db_unavailable" not in result


def test_synthesize_gives_distinct_message_for_db_outage():
    """This must NOT be confused with the 'SOPs don't cover this'
    message - a user seeing the wrong one during a real outage would
    wrongly conclude the topic isn't documented at all."""
    result = synthesize({
        "user_message": "who approves WFH",
        "retrieved_chunks": [],
        "db_unavailable": True,
    })

    assert "database connectivity" in result["answer"].lower()
    assert "don't have information" not in result["answer"]
    assert result["confidence"] == "Low"


def test_synthesize_gives_normal_no_answer_message_when_db_is_fine():
    """Regression guard: the new db_unavailable branch must not
    accidentally fire for the ordinary 'genuinely no matching content'
    case."""
    result = synthesize({
        "user_message": "what is the maternity leave policy",
        "retrieved_chunks": [],
    })

    assert "don't have information" in result["answer"]
    assert "database connectivity" not in result["answer"].lower()

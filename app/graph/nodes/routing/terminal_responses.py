"""
Terminal nodes for intention outcomes that answer directly, without
touching retrieval at all - matching Section 3's "no DB call" paths.

These are intentionally simple placeholders for chitchat/meta/
actionable/escalation - real product wording can be refined later,
but the IMPORTANT thing proven here is that these paths never call
retrieval, never call embed_query, never touch Postgres - that's the
actual architectural guarantee being tested, not the exact copy.
"""
from app.graph.state import GraphState


def answer_chitchat(state: GraphState) -> dict:
    return {
        "answer": "Hi! I can help answer questions about company SOPs - "
        "leave, WFH, hiring, testing processes, and more. What would "
        "you like to know?",
        "confidence": "High",
        "sources": [],
    }


def answer_meta(state: GraphState) -> dict:
    return {
        "answer": "I answer questions about Meldep's SOPs using the "
        "actual documents as my source - I always cite where an answer "
        "comes from, and I'll tell you clearly if I'm not confident "
        "rather than guess.",
        "confidence": "High",
        "sources": [],
    }


def answer_actionable(state: GraphState) -> dict:
    return {
        "answer": "I can explain the process for that, but I can't "
        "perform actions in Meldep on your behalf yet - you'll need to "
        "do that step yourself in Meldep. Ask me how the process works "
        "and I can walk you through it.",
        "confidence": "High",
        "sources": [],
    }


def route_to_escalation(state: GraphState) -> dict:
    return {
        "answer": "This sounds like something that needs a person, not "
        "a chatbot - please reach out to HR or your Team Lead directly "
        "for this.",
        "confidence": "High",
        "sources": [],
    }

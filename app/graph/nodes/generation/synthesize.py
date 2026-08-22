"""
Synthesis node - writes the final answer from the top reranked chunks,
with citations, and computes a confidence level from real retrieval
signals.

Circuit breaker: if Gemini is down when synthesis is attempted, this
returns a graceful, honest fallback answer - the retrieved sources
themselves, without an AI-written summary - instead of raising and
letting a raw error surface in the UI (the exact "Something went
wrong answering that question" failure mode this is meant to prevent).
The user still gets something useful: real source links, just not an
AI-composed paragraph. Confidence is explicitly "Low" in this case,
since no synthesis/verification happened.
"""
from app.graph.state import GraphState
from app.llm.gemini import DEFAULT_MODEL, generate_content_safe
from app.reliability.circuit_breaker import CircuitOpenError

HIGH_CONFIDENCE_THRESHOLD = 0.6
MEDIUM_CONFIDENCE_THRESHOLD = 0.4

SYNTHESIS_PROMPT = """Answer the user's question using ONLY the information in the excerpts below. Do not use any outside knowledge. If the excerpts don't actually answer the question, say so plainly rather than guessing.

When you state a specific fact (a number, an approver's name, a rule), make sure it's exactly what the excerpt says - don't soften or generalize an exact requirement (for example, if an excerpt says two people must BOTH approve something, don't write that either one alone can).

Question: {question}

Excerpts:
{excerpts}

Write a clear, direct answer.
"""


def _format_excerpts(chunks) -> str:
    parts = []
    for i, c in enumerate(chunks):
        parts.append(
            f"[{i + 1}] From \"{c.document_title}\" ({c.heading_path}):\n{c.content}"
        )
    return "\n\n".join(parts)


def _compute_confidence(chunks) -> str:
    if not chunks:
        return "Low"
    top_score = chunks[0].combined_score
    if top_score >= HIGH_CONFIDENCE_THRESHOLD:
        return "High"
    if top_score >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "Medium"
    return "Low"


def _dedupe_sources(chunks) -> list[dict]:
    seen = set()
    sources = []
    for c in chunks:
        key = (c.document_title, c.document_version, c.heading_path)
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "document_title": c.document_title,
            "document_version": c.document_version,
            "heading_path": c.heading_path,
        })
    return sources


def synthesize(state: GraphState) -> dict:
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return {
            "answer": "I don't have information on this in the SOPs I "
            "can search. You may want to check with HR or your Team "
            "Lead directly.",
            "confidence": "Low",
            "sources": [],
        }

    query = state.get("resolved_query") or state["user_message"]
    excerpts = _format_excerpts(chunks)
    sources = _dedupe_sources(chunks)

    try:
        response = generate_content_safe(
            model=DEFAULT_MODEL,
            contents=SYNTHESIS_PROMPT.format(question=query, excerpts=excerpts),
        )
    except CircuitOpenError:
        # Gemini is down. Graceful degradation: give the user the raw
        # sources directly rather than a raw error - they can click
        # through and read the SOP themselves. This is deliberately
        # honest about what happened, not a fake-confident answer.
        source_list = "\n".join(
            f"- {s['document_title']} ({s['heading_path']})" for s in sources
        )
        return {
            "answer": "I'm having trouble generating a written answer "
            "right now, but here are the SOP sections most relevant to "
            "your question - you can check them directly:\n" + source_list,
            "confidence": "Low",
            "sources": sources,
        }

    confidence = _compute_confidence(chunks)

    return {
        "answer": response.text,
        "confidence": confidence,
        "sources": sources,
    }

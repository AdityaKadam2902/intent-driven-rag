"""
Structured fact extraction - Section 2 of the architecture doc.

Pulls exact, verifiable facts out of SOP content (numbers, approvers,
conditions) into a small structured table, separate from the fuzzy
chunk-based retrieval. This is what lets a question like "who approves
a Team Lead's leave" get looked up directly instead of relying on an
LLM correctly re-reading a paragraph every single time.

Runs once per document at ingestion time, not per query - so the extra
LLM call cost is a one-time ingestion cost, not a per-question cost.
"""
from pydantic import BaseModel, Field

from app.llm.gemini import DEFAULT_MODEL, get_client
from google.genai import types


class ExtractedFact(BaseModel):
    policy: str = Field(
        description="Short snake_case identifier for what this fact is about, "
        "e.g. 'leave_approver_team_lead' or 'wfh_intern_allowed'"
    )
    value: str = Field(
        description="The exact fact, copied precisely from the source - "
        "do not paraphrase numbers, names, or logical connectors like "
        "AND/OR. If the source says 'AND' preserve 'AND', not 'or'."
    )
    condition: str | None = Field(
        default=None,
        description="Any condition or exception attached to this fact, "
        "or null if there is none. Preserve exceptions exactly.",
    )
    source_heading: str = Field(
        description="The heading or section this fact was found under"
    )


class FactExtractionResult(BaseModel):
    facts: list[ExtractedFact]


EXTRACTION_PROMPT = """You are extracting precise, verifiable facts from a company SOP document.

Extract every factual rule you find - especially:
- Exact numbers (days, counts, percentages, versions)
- Named approvers or decision-makers, and whether multiple people are
  required together (AND) or any one of them is sufficient (OR)
- Eligibility rules and their exceptions
- Required steps, tools, or channels for a process

Rules:
- Copy values EXACTLY as written. Do not paraphrase. Do not simplify
  "X AND Y" to "X or Y" or vice versa - this changes the meaning and is
  a serious error.
- If a fact has a condition or exception attached, capture it in full,
  don't drop it.
- Only extract facts that are actually stated. Do not infer or guess
  facts that aren't explicitly present.
- If nothing fact-like is present in a section (e.g. it's purely
  descriptive prose with no rule, number, or named approver), it is
  fine to extract zero facts from it.

Document title: {doc_title}

Content to extract facts from:
---
{content}
---
"""


def extract_facts(content: str, doc_title: str = "") -> list[ExtractedFact]:
    """Runs one Gemini call against a chunk (or full document body) and
    returns a list of structured facts. Returns an empty list if the
    content has no fact-like statements - that's a valid outcome, not
    an error."""
    client = get_client()
    prompt = EXTRACTION_PROMPT.format(doc_title=doc_title, content=content)

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FactExtractionResult,
            temperature=0.0,  # deterministic - this is extraction, not creative writing
        ),
    )

    result: FactExtractionResult = response.parsed
    return result.facts


def extract_facts_for_chunks(chunks, doc_title: str = "") -> dict[int, list[ExtractedFact]]:
    """Runs extraction per chunk rather than on the whole document at
    once. Slightly more LLM calls, but keeps facts traceable back to
    the exact chunk (and therefore exact heading_path) they came from -
    needed for citations later.

    Paced with a short delay between calls to stay under the free-tier
    rate limit (15 requests/minute for gemini-3.1-flash-lite as of this
    writing - see https://ai.google.dev/gemini-api/docs/rate-limits).
    This pacing is for local/manual testing only; production ingestion
    should batch requests properly instead of a flat sleep."""
    import time

    results = {}
    for i, chunk in enumerate(chunks):
        if i > 0:
            time.sleep(4.5)  # ~13 requests/minute, safely under the 15/min free-tier cap
        facts = extract_facts(chunk.content, doc_title=doc_title)
        if facts:
            results[chunk.chunk_index] = facts
    return results

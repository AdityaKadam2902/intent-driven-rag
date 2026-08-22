"""
Entity tagging - Section 5 of the architecture doc.

Redesigned after live-testing showed two problems with a fully free-form
approach:
  1. Casing drift ("Intern" vs "intern") would silently break exact-match
     cross-document search.
  2. Topic labels drifted per chunk instead of staying consistent
     (e.g. "change management", "incident reporting", "RCA" each only
     ever appeared once - no reuse, so they're not useful for search).

Fix, in two parts:
  - Coarse topic tags (category, sub_category) are pulled directly from
    the document's own Meldep metadata - zero AI cost, zero drift risk,
    always present. This is more authoritative than anything an LLM
    would invent anyway, since it's Meldep's own real taxonomy.
  - Fine-grained topic tags are still AI-chosen, but constrained to a
    fixed vocabulary (FINE_GRAINED_TOPICS below), built from what
    actually appeared across real live-tested SOP samples, consolidated
    to remove near-duplicates. The AI can choose zero, one, or a few
    from this list - it cannot invent new ones.
  - People and roles remain free-form (names and role titles vary too
    much for a fixed list to make sense), but are now explicitly
    normalized to consistent casing after extraction.
"""
import time

from pydantic import BaseModel, Field
from google.genai import types

from app.llm.gemini import DEFAULT_MODEL, get_client

# Built from what was actually observed across the 3 real live-tested
# SOP samples (WFH policy, Testing Process/STLC, SDLC Module), with
# near-duplicates consolidated (e.g. "defects"/"tracking" merged into
# "defect_management"; "RCA"/"investigation" merged into
# "root_cause_analysis"). Extend this list deliberately, reviewing for
# duplicates, rather than letting it grow unconstrained per chunk.
FINE_GRAINED_TOPICS = [
    "eligibility",
    "approval",
    "escalation",
    "disciplinary",
    "documentation",
    "reporting",
    "requirements",
    "planning",
    "execution",
    "defect_management",
    "regression",
    "change_management",
    "traceability",
    "root_cause_analysis",
    "tools",
    "environment_setup",
    "audit",
]


def _normalize(value: str) -> str:
    """Consistent casing so 'intern' and 'Intern' become the same tag -
    title case for readability, but the comparison/storage should treat
    these as identical regardless."""
    return value.strip().title() if value else value


class EntityTags(BaseModel):
    people: list[str] = Field(
        default_factory=list,
        description="Named individuals mentioned, e.g. 'Vikas Sir', "
        "'Sandhya Mam'. Use the exact name/title as written in the "
        "source - don't normalize or guess a full name.",
    )
    roles: list[str] = Field(
        default_factory=list,
        description="Job roles or functions mentioned, e.g. 'Team Lead', "
        "'QA', 'Intern', 'Project Manager'. Generic role names, not "
        "specific people.",
    )
    fine_grained_topics: list[str] = Field(
        default_factory=list,
        description="Zero or more topics chosen ONLY from the provided "
        "fixed list - do not invent new ones. Empty list if none apply.",
    )
    # Populated from document metadata, not by the AI - see tag_entities()
    category: str | None = None
    sub_category: str | None = None


class EntityTaggingResult(BaseModel):
    people: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    fine_grained_topics: list[str] = Field(default_factory=list)


TAGGING_PROMPT = """You are tagging a chunk of a company SOP document with the people, roles, and fine-grained topics it mentions, so it can later be found via cross-document search.

Extract:
- people: specific named individuals (e.g. "Vikas Sir")
- roles: generic job roles or functions (e.g. "Team Lead", "QA", "Intern") - not the same as people
- fine_grained_topics: choose ZERO OR MORE from this EXACT fixed list only - do not invent new topics, do not use anything not in this list:
{topic_list}

Rules:
- Use exact names/titles as written in the source for people - don't guess full names or normalize spelling.
- Roles are generic categories, not specific individuals - "Team Lead" is a role, "Vikas Sir" is a person, even if in context they might be the same individual.
- For fine_grained_topics: if nothing in the fixed list clearly applies, return an empty list. Do not force a weak match, and never return a topic that isn't in the list above.
- If people or roles categories have nothing to tag, return an empty list for them - that's valid, not an error.

Document title: {doc_title}
Heading context: {heading_path}

Content to tag:
---
{content}
---
"""


def tag_entities(
    content: str,
    doc_title: str = "",
    heading_path: str = "",
    category: str | None = None,
    sub_category: str | None = None,
) -> EntityTags:
    """Runs one Gemini call for people/roles/fine-grained topics, and
    attaches category/sub_category directly from metadata (no AI call
    needed for those - they're already known, exact, and authoritative).
    Empty lists in any AI-driven category is a valid, expected outcome,
    not an error - not every chunk mentions a person, role, or matches
    a fine-grained topic."""
    client = get_client()
    prompt = TAGGING_PROMPT.format(
        doc_title=doc_title,
        heading_path=heading_path,
        content=content,
        topic_list="\n".join(f"- {t}" for t in FINE_GRAINED_TOPICS),
    )

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EntityTaggingResult,
            temperature=0.0,  # deterministic - tagging, not creative writing
        ),
    )

    result: EntityTaggingResult = response.parsed

    # Reject any fine-grained topic the model returned that isn't
    # actually in our fixed list - a safety net in case the model
    # doesn't perfectly follow the constraint every time.
    valid_topics = [t for t in result.fine_grained_topics if t in FINE_GRAINED_TOPICS]

    return EntityTags(
        people=[_normalize(p) for p in result.people],
        roles=[_normalize(r) for r in result.roles],
        fine_grained_topics=valid_topics,
        category=category,
        sub_category=sub_category,
    )


def tag_entities_for_chunks(
    chunks, doc_title: str = "", category: str | None = None, sub_category: str | None = None
) -> dict[int, EntityTags]:
    """Runs tagging per chunk. category/sub_category come from the
    document's own metadata (parsed once, same for every chunk in this
    document) - pass them through from ParsedDocument.metadata at the
    call site.

    Paced with a short delay to stay under the free-tier rate limit (15
    requests/minute for gemini-3.1-flash-lite as of this writing) -
    see https://ai.google.dev/gemini-api/docs/rate-limits.
    This pacing is for local/manual testing only; production ingestion
    should batch requests properly instead of a flat sleep."""
    results = {}
    for i, chunk in enumerate(chunks):
        if i > 0:
            time.sleep(4.5)  # ~13 requests/minute, safely under the 15/min free-tier cap
        tags = tag_entities(
            chunk.content,
            doc_title=doc_title,
            heading_path=chunk.heading_path,
            category=category,
            sub_category=sub_category,
        )
        results[chunk.chunk_index] = tags
    return results

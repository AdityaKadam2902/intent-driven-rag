"""
Splits a parsed SOP body into rule-safe chunks.

Core rule from the architecture docs: never split a heading's bullet list
away from its lead-in sentence, and never split a numbered FAQ item's
question away from its answer.

Detects structure generically (blank-line-separated blocks, heading vs.
list vs. lead-in-sentence), not by hardcoding section names like
"Purpose" or "Entry Criteria" - so it works on SOPs it hasn't seen.
"""
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    heading_path: str      # e.g. "Test Planning > Test Plan Preparation"
    content: str
    chunk_index: int = 0


HEADING_LINE = re.compile(r"^[A-Z][A-Za-z0-9 /()\-]{2,60}$")
NUMBERED_ITEM = re.compile(r"^\d+\.\s+.+")
BULLET_LINE = re.compile(r"^-\s+.+")
LEADIN_SENTENCE = re.compile(r".+:\s*$")


def _is_heading(line: str, next_line: str) -> bool:
    """A line is treated as a heading if it's short, title-cased-ish,
    has no ending punctuation, and is NOT itself a bullet or numbered item."""
    line = line.strip()
    if not line or BULLET_LINE.match(line) or NUMBERED_ITEM.match(line):
        return False
    if line.endswith((".", ",", ":", "?")):
        return False
    if len(line.split()) > 8:
        return False
    return bool(HEADING_LINE.match(line))


def _split_into_blocks(body: str) -> list[str]:
    """Split on blank lines into raw blocks first."""
    raw_blocks = re.split(r"\n\s*\n", body.strip())
    return [b.strip() for b in raw_blocks if b.strip()]


def chunk_document(body: str, doc_title: str = "") -> list[Chunk]:
    blocks = _split_into_blocks(body)
    chunks: list[Chunk] = []
    # pending_parent holds a lone top-level heading that hasn't yet been
    # "claimed" by its one immediate child block. Only the SINGLE next
    # heading+content block nests under it - this is a deliberate,
    # predictable limit, not a guess: plain text with no indentation gives
    # no reliable way to know how many children a grouping heading
    # actually has, so we don't try to infer it. Content is never dropped
    # or merged wrong; at worst a block that should be nested 2 levels
    # deep is labelled 1 level deep instead - a labelling limitation,
    # not a content bug. Numbered-item sequences are the one exception:
    # a numbered list continuing is NOT ambiguous, so the parent label
    # persists across the whole run (see Case 3 below).
    pending_parent: str | None = None

    i = 0
    while i < len(blocks):
        block = blocks[i]
        lines = block.split("\n")
        first_line = lines[0].strip()

        # Case 1: a lone heading line on its own - becomes a candidate
        # parent for exactly the next block, then is cleared.
        if len(lines) == 1 and _is_heading(first_line, ""):
            pending_parent = first_line
            i += 1
            continue

        # Case 2: heading + its own content in the same block
        if _is_heading(first_line, lines[1] if len(lines) > 1 else ""):
            heading = first_line
            rest = "\n".join(lines[1:]).strip()
            if pending_parent:
                path = f"{pending_parent} > {heading}"
                pending_parent = None  # claimed - next block is top-level again
            else:
                path = heading
            if rest:
                chunks.append(Chunk(heading_path=path, content=f"{heading}\n{rest}"))
            i += 1
            continue

        # Case 3: numbered FAQ items - Q/A pair is its own rule unit.
        # Unlike heading nesting (Case 2), a numbered sequence continuing
        # is NOT ambiguous - item 2 clearly still belongs to the same FAQ
        # group as item 1, so the parent label persists across the whole
        # run of numbered items, not just the first one.
        if NUMBERED_ITEM.match(first_line):
            path = pending_parent or doc_title
            chunks.append(Chunk(heading_path=path, content=block))
            # do NOT clear pending_parent here - let it carry to the next
            # numbered item too. It only clears once something else
            # (a new heading) claims or overrides it.
            i += 1
            continue

        # Case 4: bare paragraph, no heading detected
        path = pending_parent or doc_title
        chunks.append(Chunk(heading_path=path, content=block))
        pending_parent = None
        i += 1

    for idx, c in enumerate(chunks):
        c.chunk_index = idx

    return chunks


def _merge_orphan_leadins(chunks: list[Chunk]) -> list[Chunk]:
    """Safety net: if a chunk's content ends with a lead-in sentence like
    '...containing:' and has no bullets after it, merge it forward into
    the next chunk. Catches heading/list splits the block-splitter missed."""
    merged: list[Chunk] = []
    i = 0
    while i < len(chunks):
        c = chunks[i]
        last_line = c.content.strip().split("\n")[-1]
        if LEADIN_SENTENCE.match(last_line) and i + 1 < len(chunks):
            nxt = chunks[i + 1]
            combined = Chunk(
                heading_path=c.heading_path,
                content=c.content + "\n" + nxt.content,
            )
            merged.append(combined)
            i += 2
            continue
        merged.append(c)
        i += 1
    for idx, c in enumerate(merged):
        c.chunk_index = idx
    return merged


def chunk_sop(body: str, doc_title: str = "") -> list[Chunk]:
    """Public entry point: parsed SOP body in, rule-safe chunks out."""
    chunks = chunk_document(body, doc_title)
    chunks = _merge_orphan_leadins(chunks)
    return chunks

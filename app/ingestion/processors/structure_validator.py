"""
Structure-quality validation - runs BEFORE chunking, not instead of it.

Real-world evidence for why this exists: the first raw .txt Sir shared
(STLC/Testing Process, before we hand-cleaned it) had headings like
"Document Repository" glued directly onto the end of the PREVIOUS
section's content, with no blank line separating them - e.g. "...The
OneDrive folder path should be mentioned in the Test Plan document.
Test Case Design Before writing test cases..." with "Test Case Design"
being a heading that got merged into the prior paragraph. Our chunker
relies on blank lines to find block boundaries; a heading with no blank
line before it becomes invisible to that logic and gets swallowed into
the wrong chunk.

Rather than build complex reconstruction logic to guess where headings
and bullets used to be (unreliable, and only ever a guess), this module
does the safer, more honest thing: detect when heading-shaped lines
appear to be merged into the middle of a block (not calling this at 100%
reliable since it's a heuristic, but demonstrably better than only
checking blank-line ratio or block length), and say so clearly - so a
human can review it - instead of silently producing bad chunks that
later cause wrong or incomplete answers.

This is a deliberate design choice: fail safely and visibly, rather
than guess confidently and be wrong - the same principle used
throughout this project (see the chunker's own docstring for the
nesting-ambiguity case, handled the same way).
"""
import re
from dataclasses import dataclass

# Bullets in real Meldep exports have appeared as both "-" and "*" -
# the raw STLC text Sir shared used "*", our hand-cleaned samples use
# "-". Recognize both so bullet-detection doesn't itself become a
# false-positive source.
BULLET_LINE = re.compile(r"^[\-\*]\s+.+", re.MULTILINE)
NUMBERED_ITEM = re.compile(r"^\d+\.\s+.+", re.MULTILINE)

# Same heading shape the chunker itself uses (kept in sync deliberately -
# see app/ingestion/processors/chunker.py HEADING_LINE) - short,
# title-case-ish, no ending punctuation.
HEADING_SHAPED_LINE = re.compile(r"^[A-Z][A-Za-z0-9 /()\-]{2,60}$")


@dataclass
class StructureQualityReport:
    is_reliable: bool
    blank_line_ratio: float          # blank lines / total lines
    has_bullets_or_numbering: bool
    longest_block_word_count: int
    mid_block_heading_count: int     # headings found glued mid-block, not at a block's start
    warnings: list[str]


def _count_mid_block_headings(body: str) -> int:
    """The most direct, precise signal: scan each blank-line-separated
    block, and count heading-shaped lines that appear AFTER the first
    line of that block. A heading-shaped line in the middle of a block
    means it got glued onto prior content with no blank-line separator
    before it - exactly the defect found in the real raw STLC text
    (e.g. "Document Repository" appearing mid-paragraph)."""
    blocks = re.split(r"\n\s*\n", body.strip())
    count = 0
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        for line in lines[1:]:  # skip the first line - only check the rest
            is_bullet_or_numbered = BULLET_LINE.match(line) or NUMBERED_ITEM.match(line)
            if not is_bullet_or_numbered and HEADING_SHAPED_LINE.match(line):
                count += 1
    return count


def assess_structure_quality(
    body: str,
    min_blank_line_ratio: float = 0.05,
    max_reasonable_block_words: int = 150,
    max_mid_block_headings: int = 0,
) -> StructureQualityReport:
    """Checks whether a document's plain-text structure looks reliable
    enough for the chunker's blank-line/bullet-based heuristics to work
    correctly. Returns a report, not a boolean - callers decide what to
    do (skip ingestion, log a warning, flag for manual review, etc.)
    rather than this function making that call unilaterally.
    """
    warnings = []

    lines = body.split("\n")
    total_lines = len(lines) or 1
    blank_lines = sum(1 for line in lines if line.strip() == "")
    blank_line_ratio = blank_lines / total_lines

    has_bullets = bool(BULLET_LINE.search(body))
    has_numbers = bool(NUMBERED_ITEM.search(body))
    has_bullets_or_numbering = has_bullets or has_numbers

    blocks = re.split(r"\n\s*\n", body.strip())
    word_counts = [len(b.split()) for b in blocks if b.strip()]
    longest_block_word_count = max(word_counts) if word_counts else 0

    mid_block_heading_count = _count_mid_block_headings(body)

    if mid_block_heading_count > max_mid_block_headings:
        warnings.append(
            f"Found {mid_block_heading_count} heading-shaped line(s) "
            f"appearing in the middle of a block instead of starting "
            f"their own block - this means a heading got glued onto "
            f"the end of the previous section with no blank line "
            f"before it, exactly the defect found in the original raw "
            f"STLC export before it was hand-cleaned. The chunker will "
            f"likely swallow this heading into the wrong section."
        )

    if blank_line_ratio < min_blank_line_ratio:
        warnings.append(
            f"Very few blank lines ({blank_line_ratio:.1%} of lines) - "
            f"headings and sections may be flattened together without "
            f"separation."
        )

    if not has_bullets_or_numbering and len(body.split()) > 100:
        warnings.append(
            "No bullet points or numbered items detected in a "
            "substantial document - list items may be flattened into "
            "run-on sentences, making it hard to tell where one item "
            "ends and the next begins."
        )

    if longest_block_word_count > max_reasonable_block_words:
        warnings.append(
            f"At least one block between blank lines is unusually long "
            f"({longest_block_word_count} words) - likely multiple "
            f"headings/sections got merged into a single block with no "
            f"separator between them."
        )

    is_reliable = len(warnings) == 0

    return StructureQualityReport(
        is_reliable=is_reliable,
        blank_line_ratio=blank_line_ratio,
        has_bullets_or_numbering=has_bullets_or_numbering,
        longest_block_word_count=longest_block_word_count,
        mid_block_heading_count=mid_block_heading_count,
        warnings=warnings,
    )

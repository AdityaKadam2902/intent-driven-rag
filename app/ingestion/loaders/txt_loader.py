"""
Today's one real loader: reads SOP data from .txt files, one file per SOP,
metadata block at the top (Key: Value lines) followed by a blank line then
the body content.

This is confirmed as the actual format Meldep is providing (loaded via
n8n on their side, out of scope for this codebase). If that mechanism
changes later, a new loader implementing BaseLoader gets added - this
file and its logic don't need to change, and nothing in app/ingestion/
processors/ or beyond needs to change either.
"""
import os
import re

from app.ingestion.loaders.base import BaseLoader, ParsedDocument

# A metadata line looks like "Key: Value" or "Key words: Value" - short
# key, colon, then value. The metadata block ends at the first blank line.
METADATA_LINE = re.compile(r"^([A-Za-z][A-Za-z ]{1,30}):\s*(.+)$")


class TxtLoader(BaseLoader):
    def load(self, source: str) -> ParsedDocument:
        with open(source, "r", encoding="utf-8") as f:
            raw = f.read()
        return self._parse(raw, source_ref=source)

    def load_all(self, source_dir: str) -> list[ParsedDocument]:
        docs = []
        for filename in sorted(os.listdir(source_dir)):
            if filename.endswith(".txt"):
                path = os.path.join(source_dir, filename)
                docs.append(self.load(path))
        return docs

    @staticmethod
    def _parse(raw: str, source_ref: str) -> ParsedDocument:
        lines = raw.split("\n")
        metadata: dict = {}
        body_start_idx = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            if stripped == "":
                # blank line ends the metadata block, but only if we've
                # already captured at least one metadata field - a
                # leading blank line alone shouldn't count
                if metadata:
                    body_start_idx = i + 1
                    break
                else:
                    continue

            match = METADATA_LINE.match(stripped)
            if match:
                key, value = match.group(1).strip(), match.group(2).strip()
                metadata[key] = value
                body_start_idx = i + 1
            else:
                # first non-metadata-shaped line - metadata block is
                # over, even without a blank line separator (defensive
                # fallback for files with no blank line after metadata)
                body_start_idx = i
                break

        body = "\n".join(lines[body_start_idx:]).strip()

        return ParsedDocument(metadata=metadata, body=body, source_ref=source_ref)

"""
Loader interface - the contract between "however Meldep's data actually
arrives" and everything downstream in the pipeline.

Meldep's team is deciding the real delivery mechanism separately (webhook,
polling, direct DB access - not yet decided). Whatever it turns out to be,
it should produce a ParsedDocument and nothing downstream should need to
change. Today there is exactly one implementation, TxtLoader, but this
interface exists so a future loader is a new file, not a rewrite.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ParsedDocument:
    metadata: dict = field(default_factory=dict)
    body: str = ""
    source_ref: str = ""   # file path today; could be a webhook payload id,
                            # an API response id, etc. for future loaders


class BaseLoader(ABC):
    """Any loader - .txt file, future webhook receiver, future API
    puller - must implement this and return ParsedDocument objects."""

    @abstractmethod
    def load(self, source: str) -> ParsedDocument:
        """Load a single document from the given source reference."""
        raise NotImplementedError

    @abstractmethod
    def load_all(self, source_dir: str) -> list[ParsedDocument]:
        """Bulk-load every available document - used for the first-time
        load of all existing SOPs."""
        raise NotImplementedError

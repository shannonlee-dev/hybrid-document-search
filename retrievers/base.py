"""Minimal shared contract; no index lifecycle or algorithm is prescribed."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
    """One hit: stable string ID, 1-based rank, and retriever-specific score.

    Results are ordered best-first. Scores are not comparable across retrievers.
    """

    document_id: str
    rank: int
    score: float
    title: str | None = None
    snippet: str | None = None


class Retriever(Protocol):
    """Return at most top_k hits for a query; callers supply a positive top_k."""

    def search(self, query: str, top_k: int) -> list[SearchResult]: ...

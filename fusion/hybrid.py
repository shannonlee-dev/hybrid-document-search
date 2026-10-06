"""Role 3: Sparse/dense retrieval combined with RRF; algorithm implementation is deferred."""

from retrievers.base import SearchResult


class HybridRetriever:
    """Placeholder implementing the shared search signature."""

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return ranked hits once the retrieval implementation is available."""
        raise NotImplementedError("TODO (Role 3): implement HybridRetriever")

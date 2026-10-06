"""Role 1: BM25 sparse retrieval using bm25s; algorithm implementation is deferred."""

from retrievers.base import SearchResult


class BM25Retriever:
    """Placeholder implementing the shared search signature."""

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return ranked hits once the retrieval implementation is available."""
        raise NotImplementedError("TODO (Role 1): implement BM25Retriever")

"""Role 2: Sentence Transformer and FAISS dense retrieval; algorithm implementation is deferred."""

from retrievers.base import SearchResult


class DenseRetriever:
    """Placeholder implementing the shared search signature."""

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return ranked hits once the retrieval implementation is available."""
        raise NotImplementedError("TODO (Role 2): implement DenseRetriever")

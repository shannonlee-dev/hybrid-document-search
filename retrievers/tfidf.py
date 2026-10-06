"""Role 1: TF-IDF sparse retrieval using scikit-learn; algorithm implementation is deferred."""

from retrievers.base import SearchResult


class TfidfRetriever:
    """Placeholder implementing the shared search signature."""

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return ranked hits once the retrieval implementation is available."""
        raise NotImplementedError("TODO (Role 1): implement TfidfRetriever")

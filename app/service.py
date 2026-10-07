"""Role 3: route search requests to configured retrievers.

SearchService only selects a retriever and passes results through unchanged.
Retrieval and fusion stay in their own modules.
"""

import logging
from collections.abc import Mapping
from pathlib import Path

from app.schemas import RetrievalMethod
from data.loader import load_prepared_documents
from fusion.hybrid import HybridRetriever
from retrievers.base import Retriever, SearchResult

logger = logging.getLogger(__name__)

PREPARE_HINT = "uv run --extra sparse python -m scripts.prepare_dataset --download"


class MethodUnavailableError(Exception):
    """The requested retrieval method has no configured retriever."""

    def __init__(self, method: RetrievalMethod, reason: str) -> None:
        super().__init__(f"{method.value} search is unavailable: {reason}")
        self.method = method
        self.reason = reason


class SearchService:
    """Hold initialized retrievers and dispatch one search call to one of them."""

    def __init__(
        self,
        retrievers: Mapping[RetrievalMethod, Retriever],
        unavailable: Mapping[RetrievalMethod, str] | None = None,
    ) -> None:
        self._retrievers = dict(retrievers)
        self._unavailable = dict(unavailable or {})

    def availability(self) -> dict[RetrievalMethod, str | None]:
        """Return None for each available method, otherwise the reason it is not."""
        return {
            method: None
            if method in self._retrievers
            else self._unavailable.get(method, "not configured")
            for method in RetrievalMethod
        }

    def search(
        self, query: str, method: RetrievalMethod, top_k: int
    ) -> list[SearchResult]:
        """Search with one method and return at most ``top_k`` ranked results."""
        retriever = self._retrievers.get(method)
        if retriever is None:
            reason = self._unavailable.get(method, "not configured")
            raise MethodUnavailableError(method, reason)
        return list(retriever.search(query, top_k)[:top_k])


def build_search_service(corpus_path: Path) -> SearchService:
    """Load the corpus once and construct every retriever that can run now.

    Failures do not raise. The service starts with the affected methods marked
    unavailable, so the API process stays up and reports why search is not ready.
    """
    try:
        documents = load_prepared_documents(corpus_path)
    except FileNotFoundError:
        logger.error("corpus not found: %s", corpus_path)
        reason = f"corpus is missing. Prepare it with: {PREPARE_HINT}"
        return _all_unavailable(reason)
    except (OSError, ValueError) as exc:
        logger.error("corpus could not be loaded: %s", exc)
        return _all_unavailable("corpus is invalid; see server logs.")

    retrievers: dict[RetrievalMethod, Retriever] = {}
    unavailable: dict[RetrievalMethod, str] = {}

    try:
        from retrievers.tfidf import TfidfRetriever

        retrievers[RetrievalMethod.TFIDF] = TfidfRetriever(documents)
    except ModuleNotFoundError as exc:
        if exc.name != "sklearn" and not (exc.name or "").startswith("sklearn."):
            raise
        unavailable[RetrievalMethod.TFIDF] = (
            "scikit-learn is missing. Install it with: uv sync --extra sparse"
        )
    except ValueError:
        logger.exception("TF-IDF index could not be built")
        unavailable[RetrievalMethod.TFIDF] = (
            "index could not be built; see server logs."
        )

    # BM25 and Dense are not merged yet. Register each one here after its Role merges.
    unavailable[RetrievalMethod.BM25] = "BM25 retriever is not merged yet."
    unavailable[RetrievalMethod.DENSE] = "Dense retriever is not merged yet."
    # Hybrid is BM25 + Dense + RRF only. There is no TF-IDF fallback.
    missing = [
        name
        for name, method in (
            ("BM25", RetrievalMethod.BM25),
            ("Dense", RetrievalMethod.DENSE),
        )
        if method not in retrievers
    ]
    if missing:
        unavailable[RetrievalMethod.HYBRID] = (
            f"requires {' and '.join(missing)}, which is not available."
        )
    else:
        retrievers[RetrievalMethod.HYBRID] = HybridRetriever(
            retrievers[RetrievalMethod.BM25], retrievers[RetrievalMethod.DENSE]
        )

    for method, reason in unavailable.items():
        logger.warning("%s search unavailable: %s", method.value, reason)
    return SearchService(retrievers, unavailable)


def _all_unavailable(reason: str) -> SearchService:
    return SearchService({}, {method: reason for method in RetrievalMethod})

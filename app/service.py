"""Role 3: route search requests to configured retrievers.

SearchService only selects a retriever and passes results through unchanged.
Retrieval and fusion stay in their own modules.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.schemas import RetrievalMethod
from data.loader import Document, load_prepared_documents
from fusion.hybrid import HybridRetriever
from retrievers.base import Retriever, SearchResult

logger = logging.getLogger(__name__)

PREPARE_HINT = "uv run --extra sparse python -m scripts.prepare_dataset --download"
DENSE_BUILD_HINT = (
    "uv run --extra dense python -m scripts.build_index "
    "--corpus data/processed/corpus.jsonl --index <DENSE_INDEX_DIR>"
)


class MethodState(StrEnum):
    """Why a method can or cannot search. Each value maps to one failure stage."""

    AVAILABLE = "available"
    NOT_IMPLEMENTED = "not_implemented"  # the retriever module is still a placeholder
    NOT_CONFIGURED = "not_configured"  # no setting points at this method's artifact
    CORPUS_UNAVAILABLE = "corpus_unavailable"  # corpus missing or invalid
    DEPENDENCY_MISSING = "dependency_missing"  # optional Python package not installed
    INDEX_MISSING = "index_missing"  # configured index artifact not on disk
    LOAD_FAILED = "load_failed"  # index or model failed to load or does not match
    UPSTREAM_UNAVAILABLE = (
        "upstream_unavailable"  # a component this method needs is down
    )


@dataclass(frozen=True)
class MethodStatus:
    state: MethodState
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.state is MethodState.AVAILABLE


AVAILABLE = MethodStatus(MethodState.AVAILABLE)


class MethodUnavailableError(Exception):
    """The requested retrieval method cannot search. Carries the reason."""

    def __init__(self, method: RetrievalMethod, status: MethodStatus) -> None:
        reason = status.reason or status.state.value
        super().__init__(f"{method.value} search is unavailable: {reason}")
        self.method = method
        self.status = status


class SearchService:
    """Hold initialized retrievers and dispatch one search call to one of them."""

    def __init__(
        self,
        retrievers: Mapping[RetrievalMethod, Retriever],
        statuses: Mapping[RetrievalMethod, MethodStatus] | None = None,
    ) -> None:
        self._retrievers = dict(retrievers)
        self._statuses = dict(statuses or {})

    def availability(self) -> dict[RetrievalMethod, MethodStatus]:
        """Return the status of every method. Available means initialized."""
        return {method: self._status(method) for method in RetrievalMethod}

    def search(
        self, query: str, method: RetrievalMethod, top_k: int
    ) -> list[SearchResult]:
        """Search with one method and return at most ``top_k`` ranked results."""
        retriever = self._retrievers.get(method)
        if retriever is None:
            raise MethodUnavailableError(method, self._status(method))
        return list(retriever.search(query, top_k)[:top_k])

    def _status(self, method: RetrievalMethod) -> MethodStatus:
        if method in self._retrievers:
            return AVAILABLE
        return self._statuses.get(
            method, MethodStatus(MethodState.NOT_CONFIGURED, "not configured")
        )


def build_search_service(
    corpus_path: Path,
    *,
    dense_index_path: Path | None = None,
    dense_device: str | None = None,
) -> SearchService:
    """Load the corpus once and initialize every retriever that can run.

    Failures do not raise. Each affected method gets a MethodStatus that says why,
    so the API process stays up and reports the state. Paths are logged, not returned.
    """
    try:
        documents = load_prepared_documents(corpus_path)
    except FileNotFoundError:
        logger.error("corpus not found: %s", corpus_path)
        return _all_unavailable(
            MethodState.CORPUS_UNAVAILABLE,
            f"corpus is missing. Prepare it with: {PREPARE_HINT}",
        )
    except (OSError, ValueError) as exc:
        logger.error("corpus could not be loaded: %s", exc)
        return _all_unavailable(
            MethodState.CORPUS_UNAVAILABLE, "corpus is invalid; see server logs."
        )

    retrievers: dict[RetrievalMethod, Retriever] = {}
    statuses: dict[RetrievalMethod, MethodStatus] = {}

    def register(method: RetrievalMethod, outcome: Retriever | MethodStatus) -> None:
        if isinstance(outcome, MethodStatus):
            statuses[method] = outcome
        else:
            retrievers[method] = outcome

    register(RetrievalMethod.TFIDF, _tfidf(documents))
    register(RetrievalMethod.BM25, _bm25(documents))
    register(
        RetrievalMethod.DENSE,
        _dense(documents, dense_index_path, dense_device),
    )

    bm25 = retrievers.get(RetrievalMethod.BM25)
    dense = retrievers.get(RetrievalMethod.DENSE)
    if bm25 is not None and dense is not None:
        # Hybrid is BM25 + Dense + RRF. There is no TF-IDF fallback.
        retrievers[RetrievalMethod.HYBRID] = HybridRetriever(bm25, dense)
    else:
        missing = [
            name
            for name, retriever in (("BM25", bm25), ("Dense", dense))
            if retriever is None
        ]
        statuses[RetrievalMethod.HYBRID] = MethodStatus(
            MethodState.UPSTREAM_UNAVAILABLE,
            f"requires {' and '.join(missing)}, which is not available.",
        )

    service = SearchService(retrievers, statuses)
    for method, status in service.availability().items():
        if status.available:
            logger.info("%s search: available", method.value)
        else:
            logger.warning(
                "%s search: %s (%s)", method.value, status.state.value, status.reason
            )
    return service


def _is_placeholder(cls: type) -> bool:
    """Role-owned placeholders define no constructor; a merged implementation does."""
    return "__init__" not in vars(cls)


def _tfidf(documents: list[Document]) -> Retriever | MethodStatus:
    try:
        from retrievers.tfidf import TfidfRetriever
    except ModuleNotFoundError as exc:
        if exc.name != "sklearn" and not (exc.name or "").startswith("sklearn."):
            raise
        return MethodStatus(
            MethodState.DEPENDENCY_MISSING,
            "scikit-learn is missing. Install it with: uv sync --extra sparse",
        )
    try:
        return TfidfRetriever(documents)
    except ValueError:
        logger.exception("TF-IDF index could not be built")
        return MethodStatus(
            MethodState.LOAD_FAILED, "index could not be built; see server logs."
        )


def _bm25(documents: list[Document]) -> Retriever | MethodStatus:
    try:
        from retrievers.bm25 import BM25Retriever
    except ModuleNotFoundError as exc:
        logger.error("bm25 dependency missing: %s", exc)
        return MethodStatus(
            MethodState.DEPENDENCY_MISSING,
            "BM25 dependencies are missing. Install them with: uv sync --extra sparse",
        )

    if _is_placeholder(BM25Retriever):
        return MethodStatus(
            MethodState.NOT_IMPLEMENTED, "BM25 retriever is not merged yet."
        )
    try:
        return BM25Retriever(documents)
    except ValueError:
        logger.exception("BM25 index could not be built")
        return MethodStatus(
            MethodState.LOAD_FAILED, "index could not be built; see server logs."
        )


def _dense(
    documents: list[Document],
    index_dir: Path | None,
    device: str | None,
) -> Retriever | MethodStatus:
    if index_dir is None:
        return MethodStatus(
            MethodState.NOT_CONFIGURED,
            "set DENSE_INDEX_PATH to a saved index directory.",
        )

    try:
        from retrievers.dense import DenseRetriever
    except ModuleNotFoundError as exc:
        logger.error("dense module import failed: %s", exc)
        return MethodStatus(
            MethodState.DEPENDENCY_MISSING,
            "dense dependencies are missing. Install them with: uv sync --extra dense",
        )

    if _is_placeholder(DenseRetriever):
        return MethodStatus(
            MethodState.NOT_IMPLEMENTED, "Dense retriever is not merged yet."
        )
    try:
        retriever = DenseRetriever.load(index_dir, device=device)
    except FileNotFoundError:
        logger.error("dense index not found: %s", index_dir)
        return MethodStatus(
            MethodState.INDEX_MISSING,
            f"index is missing at the configured path. Build it with: {DENSE_BUILD_HINT}",
        )
    except ModuleNotFoundError as exc:
        logger.error("dense dependency missing: %s", exc)
        return MethodStatus(
            MethodState.DEPENDENCY_MISSING,
            "dense dependencies are missing. Install them with: uv sync --extra dense",
        )
    except ValueError:
        logger.exception("dense index could not be loaded")
        return MethodStatus(
            MethodState.LOAD_FAILED, "index could not be loaded; see server logs."
        )

    if tuple(documents) != retriever.documents:
        return MethodStatus(
            MethodState.LOAD_FAILED,
            "index was built from a different corpus. Rebuild it from the current corpus.",
        )
    try:
        # Load the embedding model now, so "available" means it can actually search.
        retriever.embedder.encode_query("warm-up")
    except Exception:
        # The embedding stack raises library-specific errors; the detail is logged.
        logger.exception("dense model could not be loaded")
        return MethodStatus(
            MethodState.LOAD_FAILED,
            "embedding model could not be loaded; see server logs.",
        )
    return retriever


def _all_unavailable(state: MethodState, reason: str) -> SearchService:
    return SearchService(
        {}, {method: MethodStatus(state, reason) for method in RetrievalMethod}
    )

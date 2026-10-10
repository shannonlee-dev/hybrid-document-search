"""Initialize retrievers and route search requests by method."""

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
    """Retrieval availability and the stage preventing a method from searching."""

    AVAILABLE = "available"
    NOT_IMPLEMENTED = "not_implemented"
    NOT_CONFIGURED = "not_configured"
    CORPUS_UNAVAILABLE = "corpus_unavailable"
    DEPENDENCY_MISSING = "dependency_missing"
    INDEX_MISSING = "index_missing"
    LOAD_FAILED = "load_failed"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"


@dataclass(frozen=True)
class MethodStatus:
    state: MethodState
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.state is MethodState.AVAILABLE


AVAILABLE = MethodStatus(MethodState.AVAILABLE)


class MethodUnavailableError(Exception):
    """Search failure carrying the requested method and its availability status."""

    def __init__(self, method: RetrievalMethod, status: MethodStatus) -> None:
        reason = status.reason or status.state.value
        super().__init__(f"{method.value} search is unavailable: {reason}")
        self.method = method
        self.status = status


class SearchService:
    """Dispatch search requests to initialized retrievers."""

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

    Expected initialization failures become method statuses. Internal paths stay
    in server logs rather than availability messages.
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

    def _register(method: RetrievalMethod, outcome: Retriever | MethodStatus) -> None:
        if isinstance(outcome, MethodStatus):
            statuses[method] = outcome
        else:
            retrievers[method] = outcome

    _register(RetrievalMethod.TFIDF, _tfidf(documents))
    _register(RetrievalMethod.BM25, _bm25(documents))
    _register(
        RetrievalMethod.DENSE,
        _dense(documents, dense_index_path, dense_device),
    )

    bm25 = retrievers.get(RetrievalMethod.BM25)
    dense = retrievers.get(RetrievalMethod.DENSE)
    if bm25 is not None and dense is not None:
        # Hybrid는 BM25와 Dense가 모두 필요하며 TF-IDF로 대체할 수 없다.
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
    """Detect constructor-free retriever stubs used during integration."""
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
    except (OSError, ValueError):
        logger.exception("dense module could not be imported")
        return MethodStatus(
            MethodState.LOAD_FAILED,
            "dense module could not be imported; see server logs.",
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
    except OSError as exc:
        logger.error("dense index path is unusable: %s", exc)
        return MethodStatus(
            MethodState.LOAD_FAILED, "index path could not be read; see server logs."
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
        # 사용 가능 상태를 알리기 전에 실제 검색으로 모델과 인덱스의 호환성을 확인한다.
        retriever.search("warm-up", 1)
    except Exception:
        # 임베딩 라이브러리별 예외의 상세 원인은 서버 로그에 남긴다.
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

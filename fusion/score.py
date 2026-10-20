"""Fuse Dense and corpus-max-normalized BM25 scores over the full corpus."""

from collections.abc import Sequence
from copy import copy
from heapq import nsmallest
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING

from fusion.rrf import validate_top_k
from retrievers.base import SNIPPET_LENGTH, SearchResult

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from retrievers.bm25 import BM25Retriever
    from retrievers.dense import DenseRetriever

DEFAULT_BM25_WEIGHT = 0.05
_VECTOR_NORM_ATOL = 1e-4


class ScoreFusionRetriever:
    """Vectorized search implementation for BM25 and normalized FAISS Dense scores.

    Specialized for BM25Retriever and IndexFlatIP-backed DenseRetriever, not
    arbitrary Retriever implementations. Combine full-corpus score arrays and
    create SearchResult records only for selected hits, encoding each query once.

    Require unit document/query vectors and the same ordered corpus in both
    indexes. Snapshot document vectors at initialization; recreate this retriever
    when the corpus changes. Later FAISS mutations do not update the snapshot.
    """

    def __init__(
        self,
        sparse_retriever: "BM25Retriever",
        dense_retriever: "DenseRetriever",
        *,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
    ) -> None:
        import numpy as np

        from indexing.vectors import as_vectors

        _validate_weight(bm25_weight)
        if not dense_retriever.config.normalize_embeddings:
            raise ValueError("score fusion requires normalized Dense embeddings")
        self._documents = tuple(dense_retriever.documents)
        if sparse_retriever.documents != self._documents:
            raise ValueError("BM25 and Dense must use the same ordered corpus")
        self._vectors = as_vectors(dense_retriever.index.snapshot_vectors())
        if len(self._vectors) != len(self._documents):
            raise ValueError("Dense index count must match the corpus")
        if not np.allclose(
            np.linalg.norm(self._vectors, axis=1), 1, atol=_VECTOR_NORM_ATOL
        ):
            raise ValueError("score fusion requires normalized document vectors")
        self._vectors.flags.writeable = False
        self._embedder = dense_retriever.embedder
        self._sparse = sparse_retriever
        self._bm25_weight = float(bm25_weight)
        document_ids = [doc.document_id for doc in self._documents]
        self._tie_ranks = np.argsort(np.argsort(document_ids))

    def with_bm25_weight(self, bm25_weight: float) -> "ScoreFusionRetriever":
        """Return an independent weight setting sharing this frozen corpus snapshot.

        No vectors are copied and later FAISS mutations remain invisible. Both
        instances use the same BM25 index and embedder, as in the constructor.
        """
        _validate_weight(bm25_weight)
        variant = copy(self)
        variant._bm25_weight = float(bm25_weight)
        return variant

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Vectorized search: score all documents and return at most top_k hits.

        Resolve ties by document ID; score_fusion() is the reference implementation.
        """
        import numpy as np

        from indexing.vectors import as_vectors

        validate_top_k(top_k)
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            return []
        vector = as_vectors(self._embedder.encode_query(query), self._vectors.shape[1])
        if len(vector) != 1:
            raise ValueError("search requires exactly one query vector")
        if not np.isclose(np.linalg.norm(vector[0]), 1, atol=_VECTOR_NORM_ATOL):
            raise ValueError("score fusion requires normalized query vectors")
        scores = (self._vectors @ vector[0]).astype(np.float64)
        sparse = self._sparse.score_all(query)
        if sparse.shape != scores.shape:
            raise ValueError("BM25 scores must match the corpus shape")
        if not np.isfinite(sparse).all():
            raise ValueError("BM25 scores must be finite")
        scores = _fuse_scores(
            scores,
            sparse.astype(np.float64),
            maximum=float(np.max(sparse)),
            bm25_weight=self._bm25_weight,
        )
        if not np.isfinite(scores).all():
            raise ValueError("fused scores must be finite")
        selected = np.lexsort((self._tie_ranks, -scores))[:top_k]
        return [
            SearchResult(
                document_id=self._documents[row].document_id,
                rank=rank,
                score=float(scores[row]),
                title=self._documents[row].title,
                snippet=self._documents[row].text[:SNIPPET_LENGTH],
            )
            for rank, row in enumerate(selected, start=1)
        ]


def score_fusion(
    dense_results: Sequence[SearchResult],
    bm25_results: Sequence[SearchResult],
    top_k: int,
    *,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
) -> list[SearchResult]:
    """Fuse complete Dense scores with all nonzero BM25 scores for a query.

    Share the fusion formula with ScoreFusionRetriever.search() without NumPy.

    Require Dense scores for the whole corpus and reject duplicates within each
    source. Missing BM25 hits contribute zero; a nonpositive corpus maximum
    disables the BM25 contribution. Overlapping hits appear once with Dense
    metadata. Ties sort by document ID regardless of source rank or input order.
    Fused scores are not probabilities.

    Full-corpus inputs are required for global normalization and selection.
    """
    validate_top_k(top_k)
    _validate_weight(bm25_weight)
    dense = _index_results_by_id(dense_results)
    sparse = _index_results_by_id(bm25_results)
    if sparse.keys() - dense.keys():
        raise ValueError("BM25 documents must have a Dense score in the same corpus")
    maximum = max((result.score for result in sparse.values()), default=0.0)
    scores = {}
    for document_id, result in dense.items():
        lexical = sparse.get(document_id)
        score = _fuse_scores(
            float(result.score),
            float(lexical.score) if lexical else 0.0,
            maximum=maximum,
            bm25_weight=float(bm25_weight),
        )
        if not isfinite(score):
            raise ValueError("fused scores must be finite")
        scores[document_id] = score
    selected = nsmallest(top_k, dense, key=lambda doc_id: (-scores[doc_id], doc_id))
    return [
        SearchResult(
            document_id=doc_id,
            rank=rank,
            score=scores[doc_id],
            title=dense[doc_id].title,
            snippet=dense[doc_id].snippet,
        )
        for rank, doc_id in enumerate(selected, start=1)
    ]


def _fuse_scores[Scores: (float, "NDArray[np.float64]")](
    dense: Scores,
    sparse: Scores,
    *,
    maximum: float,
    bm25_weight: float,
) -> Scores:
    """Apply corpus-max normalization and weighting to scalars or float64 arrays.

    Callers validate inputs and supply the maximum over the full BM25 corpus.
    Array arithmetic stays vectorized; this helper does not import NumPy or
    mutate its inputs.
    """
    if maximum <= 0:
        return dense
    return dense + bm25_weight * (sparse / maximum)


def _validate_weight(weight: float) -> None:
    if (
        isinstance(weight, bool)
        or not isinstance(weight, Real)
        or not isfinite(weight)
        or weight < 0
    ):
        raise ValueError("bm25_weight must be finite and non-negative")


def _index_results_by_id(results: Sequence[SearchResult]) -> dict[str, SearchResult]:
    by_id = {}
    for result in results:
        if not isinstance(result.document_id, str) or not result.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if result.document_id in by_id:
            raise ValueError(f"duplicate document_id {result.document_id!r}")
        if (
            isinstance(result.score, bool)
            or not isinstance(result.score, Real)
            or not isfinite(result.score)
        ):
            raise ValueError("source scores must be finite numbers")
        by_id[result.document_id] = result
    return by_id

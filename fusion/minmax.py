"""Fixed alpha fusion over a Dense/BM25 top-pool union with union min-max.

This is the fusion used by the Finance query-encoder evaluation
(``evaluation.query_encoder.fixed_fusion_rankings``) exposed as a Retriever so
it can run behind the search service. It is not the service default Hybrid,
which is ``fusion.score.ScoreFusionRetriever``.
"""

from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING

from fusion.rrf import validate_top_k
from retrievers.base import SNIPPET_LENGTH, SearchResult

if TYPE_CHECKING:
    from retrievers.bm25 import BM25Retriever
    from retrievers.dense import DenseRetriever

DEFAULT_POOL_K = 100
_VECTOR_NORM_ATOL = 1e-4


class UnionMinMaxFusionRetriever:
    """Rank ``alpha*Dense + (1-alpha)*BM25`` after min-max over the union pool.

    Dense scores are FP64 inner products against an FP64 snapshot of the FAISS
    vectors. The pool is the union of each channel's top ``pool_k`` (stable
    argsort); ties in the fused score keep corpus order.
    """

    def __init__(
        self,
        sparse_retriever: "BM25Retriever",
        dense_retriever: "DenseRetriever",
        *,
        alpha: float,
        pool_k: int = DEFAULT_POOL_K,
    ) -> None:
        import numpy as np

        from indexing.vectors import as_vectors

        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, Real)
            or not isfinite(alpha)
            or not 0 <= alpha <= 1
        ):
            raise ValueError("alpha must be within [0, 1]")
        validate_top_k(pool_k)
        if not dense_retriever.config.normalize_embeddings:
            raise ValueError("fusion requires normalized Dense embeddings")
        self._documents = tuple(dense_retriever.documents)
        if sparse_retriever.documents != self._documents:
            raise ValueError("BM25 and Dense must use the same ordered corpus")
        vectors = as_vectors(dense_retriever.index.snapshot_vectors())
        if len(vectors) != len(self._documents):
            raise ValueError("Dense index count must match the corpus")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=_VECTOR_NORM_ATOL):
            raise ValueError("fusion requires normalized document vectors")
        self._vectors = vectors.astype(np.float64)
        self._vectors.flags.writeable = False
        self._embedder = dense_retriever.embedder
        self._sparse = sparse_retriever
        self.alpha = float(alpha)
        self.pool_k = pool_k

    def rank(self, dense, sparse, top_k):
        """Return (rows, fused scores) for full-corpus channel scores."""
        import numpy as np

        pool = np.union1d(
            np.argsort(-dense, kind="stable")[: self.pool_k],
            np.argsort(-sparse, kind="stable")[: self.pool_k],
        )

        def normalize(values):
            return (values - values.min()) / max(float(np.ptp(values)), 1e-30)

        fused = self.alpha * normalize(dense[pool]) + (1 - self.alpha) * normalize(
            sparse[pool]
        )
        order = np.argsort(-fused, kind="stable")[:top_k]
        return pool[order], fused[order]

    def dense_scores(self, query_vector):
        import numpy as np

        return self._vectors @ np.asarray(query_vector, dtype=np.float64)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
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
        dense = self.dense_scores(vector[0])
        sparse = np.asarray(self._sparse.score_all(query), dtype=np.float64)
        if sparse.shape != dense.shape or not np.isfinite(sparse).all():
            raise ValueError("BM25 scores must be finite and match the corpus")
        rows, scores = self.rank(dense, sparse, top_k)
        return [
            SearchResult(
                document_id=self._documents[row].document_id,
                rank=rank,
                score=float(score),
                title=self._documents[row].title,
                snippet=self._documents[row].text[:SNIPPET_LENGTH],
            )
            for rank, (row, score) in enumerate(zip(rows, scores, strict=True), start=1)
        ]

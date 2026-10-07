"""Role 3: Sparse/dense retrieval combined with RRF.

Child retrievers are injected and typed only as the shared ``Retriever`` protocol,
so this module never imports concrete TF-IDF, BM25, dense, or FAISS classes.
"""

from fusion.rrf import (
    DEFAULT_RANK_CONSTANT,
    reciprocal_rank_fusion,
    validate_rank_constant,
    validate_top_k,
)
from retrievers.base import Retriever, SearchResult


class HybridRetriever:
    """Fuse two retrievers' ranked results with RRF; implements ``Retriever``."""

    def __init__(
        self,
        sparse_retriever: Retriever,
        dense_retriever: Retriever,
        rank_constant: int = DEFAULT_RANK_CONSTANT,
    ) -> None:
        validate_rank_constant(rank_constant)
        self._sparse = sparse_retriever
        self._dense = dense_retriever
        self._rank_constant = rank_constant

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Ask both retrievers for ``top_k`` candidates and fuse by rank."""
        validate_top_k(top_k)
        sparse_results = self._sparse.search(query, top_k)
        dense_results = self._dense.search(query, top_k)
        return reciprocal_rank_fusion(
            [sparse_results, dense_results],
            top_k=top_k,
            rank_constant=self._rank_constant,
        )

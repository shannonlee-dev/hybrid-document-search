"""Fuse retrieved evidence with a local relevance judge, without relevance labels.

This DAT-style policy uses Qwen yes/no odds rather than the paper's generative
effectiveness labels. It is an explicitly scoped adaptation, not a full recipe.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import numpy as np

from data.loader import Document
from fusion.rrf import validate_top_k
from retrievers.base import SNIPPET_LENGTH, SearchResult

CANDIDATE_K = 100
NORMALIZATIONS = ("returned_minmax", "union_minmax", "union_raw", "full_raw")
_PROBABILITY_EPSILON = 1e-7


@dataclass(frozen=True)
class FixedPolicy:
    """A validation-selected Dense weight and normalization."""

    alpha: float = 0.84
    normalization: str = "union_minmax"

    def __post_init__(self):
        if (
            isinstance(self.alpha, bool)
            or not isfinite(self.alpha)
            or not 0 <= self.alpha <= 1
        ):
            raise ValueError("alpha must be finite and in [0, 1]")
        if self.normalization not in NORMALIZATIONS:
            raise ValueError("unsupported normalization")


@dataclass(frozen=True)
class ContextualRun:
    """Ranked hits and instrumentation from one uncached request."""

    results: list[SearchResult]
    alpha: float
    judged: bool
    probabilities: tuple[float, float] | None
    elapsed_ms: dict[str, float]
    top1_document_ids: tuple[str, str]


def relevance_weight(probabilities: Sequence[float]) -> float:
    """Convert [Dense, Sparse] relevance probabilities into Dense odds share."""
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (2,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("judge must return two finite probabilities in [0, 1]")
    p = np.clip(p, _PROBABILITY_EPSILON, 1 - _PROBABILITY_EPSILON)
    odds = p / (1 - p)
    return float(odds[0] / odds.sum())


def candidate_scores(dense, sparse, normalization="union_minmax"):
    """Return corpus rows and aligned normalized scores, with corpus-order ties."""
    dense, sparse = _channels(dense, sparse)
    if normalization not in NORMALIZATIONS:
        raise ValueError("unsupported normalization")
    if normalization == "full_raw":
        return np.arange(len(dense)), dense, sparse / max(float(sparse.max()), 1e-30)
    di, si = _top(dense, CANDIDATE_K), _top(sparse, CANDIDATE_K)
    pool = np.union1d(di, si)
    if normalization == "union_minmax":
        return pool, _minmax(dense[pool]), _minmax(sparse[pool])
    if normalization == "union_raw":
        return pool, dense[pool], sparse[pool] / max(float(sparse.max()), 1e-30)
    dd, ss = np.zeros(len(dense)), np.zeros(len(dense))
    dd[di], ss[si] = _minmax(dense[di]), _minmax(sparse[si])
    return pool, dd[pool], ss[pool]


def fuse_scores(dense, sparse, *, alpha, normalization="union_minmax", top_k=10):
    """Rank the top-100 union, or full corpus for the legacy normalization."""
    validate_top_k(top_k)
    FixedPolicy(alpha, normalization)
    pool, dd, ss = candidate_scores(dense, sparse, normalization)
    combined = alpha * dd + (1 - alpha) * ss
    order = _top(combined, top_k)
    return pool[order], combined[order]


class ContextualRetriever:
    """Apply the frozen disagreement policy through injectable local runtimes.

    Documents must be in lexicographic ID order, matching the BC Card cache.
    The encoder and sparse scorer are called afresh for every search. Set
    adaptive=False for the matched Fixed baseline; both share the same runtimes.
    """

    def __init__(
        self,
        documents: Sequence[Document],
        document_vectors,
        *,
        encode: Callable,
        sparse_scores: Callable,
        judge: Callable,
        fixed: FixedPolicy = FixedPolicy(),
        adaptive: bool = True,
        synchronize: Callable = lambda: None,
    ):
        self.documents = tuple(documents)
        ids = [d.document_id for d in self.documents]
        if not ids or ids != sorted(set(ids)):
            raise ValueError("documents must have unique IDs in sorted order")
        self.vectors = np.asarray(document_vectors, dtype=np.float64)
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(ids):
            raise ValueError("document vectors must match corpus rows")
        if not np.isfinite(self.vectors).all():
            raise ValueError("document vectors must be finite")
        self.encode, self.sparse_scores, self.judge = encode, sparse_scores, judge
        self.fixed, self.adaptive, self.synchronize = fixed, adaptive, synchronize

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        return self.search_detailed(query, top_k).results

    def search_detailed(self, query: str, top_k: int = 10) -> ContextualRun:
        """Return a timed request; validation and metrics run outside this method."""
        validate_top_k(top_k)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        self.synchronize()
        started = perf_counter()
        vector = np.asarray(self.encode(query), dtype=np.float64)
        if vector.shape != (self.vectors.shape[1],) or not np.isfinite(vector).all():
            raise ValueError("query vector must match the document embedding dimension")
        self.synchronize()
        encoded = perf_counter()
        dense = vector @ self.vectors.T
        sparse = np.asarray(self.sparse_scores(query), dtype=float)
        dense, sparse = _channels(dense, sparse)
        di, si = int(np.argmax(dense)), int(np.argmax(sparse))
        retrieved = perf_counter()
        judged = self.adaptive and di != si
        alpha, normalization, probabilities = (
            self.fixed.alpha,
            self.fixed.normalization,
            None,
        )
        if judged:
            probabilities = tuple(
                float(p)
                for p in self.judge(
                    query, [self.documents[di].text, self.documents[si].text]
                )
            )
            alpha = relevance_weight(probabilities)
            normalization = "union_minmax"
        self.synchronize()
        routed = perf_counter()
        order, scores = fuse_scores(
            dense, sparse, alpha=alpha, normalization=normalization, top_k=top_k
        )
        results = [
            SearchResult(
                self.documents[i].document_id,
                rank,
                float(score),
                self.documents[i].title,
                self.documents[i].text[:SNIPPET_LENGTH],
            )
            for rank, (i, score) in enumerate(zip(order, scores, strict=True), 1)
        ]
        self.synchronize()
        finished = perf_counter()
        points = [started, encoded, retrieved, routed, finished]
        elapsed = {
            name: 1000 * (b - a)
            for name, a, b in zip(
                ("encode", "retrieve", "judge", "fusion"),
                points[:-1],
                points[1:],
                strict=True,
            )
        }
        elapsed["total"] = 1000 * (finished - started)
        return ContextualRun(
            results,
            alpha,
            judged,
            probabilities,
            elapsed,
            (self.documents[di].document_id, self.documents[si].document_id),
        )


def _channels(dense, sparse):
    dense, sparse = np.asarray(dense, dtype=float), np.asarray(sparse, dtype=float)
    if dense.ndim != 1 or not dense.size or dense.shape != sparse.shape:
        raise ValueError("nonempty Dense and Sparse scores must have the same shape")
    if not np.isfinite(dense).all() or not np.isfinite(sparse).all():
        raise ValueError("scores must be finite")
    return dense, sparse


def _top(scores, k):
    k = min(k, len(scores))
    boundary = np.partition(scores, -k)[-k]
    rows = np.flatnonzero(scores >= boundary)
    return rows[np.lexsort((rows, -scores[rows]))][:k]


def _minmax(scores):
    return (scores - scores.min()) / max(float(np.ptp(scores)), 1e-30)

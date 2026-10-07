"""Role 3: Reciprocal Rank Fusion (RRF) over ranked SearchResult lists.

RRF scores a document by summing ``1 / (rank_constant + rank)`` over every source
list that contains it. Only ranks are read; each source's raw score is ignored,
because BM25, TF-IDF cosine, and dense inner-product scores are not comparable.

The fused ``SearchResult.score`` is the RRF score itself. It is not a BM25 or
cosine score, a probability, or a calibrated relevance value.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from retrievers.base import SearchResult

DEFAULT_RANK_CONSTANT = 60


@dataclass
class _Candidate:
    """Accumulator for one document across all source lists."""

    contributions: list[float]
    best_rank: int
    title: str | None
    snippet: str | None


def validate_top_k(top_k: int) -> None:
    """Require a positive int; ``bool`` is rejected even though it subclasses int."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


def validate_rank_constant(rank_constant: int) -> None:
    """Require a non-negative int so every denominator ``k + rank`` is positive."""
    if (
        isinstance(rank_constant, bool)
        or not isinstance(rank_constant, int)
        or rank_constant < 0
    ):
        raise ValueError("rank_constant must be a non-negative integer")


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[SearchResult]],
    top_k: int,
    rank_constant: int = DEFAULT_RANK_CONSTANT,
) -> list[SearchResult]:
    """Fuse ranked result lists and return at most ``top_k`` results, best first.

    Duplicates across lists are reported once, with RRF contributions summed.
    Metadata (``title``/``snippet``) comes from the single occurrence with the
    best source rank; ties go to the earlier list. Nothing is merged across
    occurrences, so no metadata is invented.

    Equal fused scores are ordered by best source rank ascending, then by
    ``document_id`` ascending. The score is summed with ``math.fsum``, so it
    does not depend on the order of the contributions.

    Raises ValueError for a non-positive ``top_k``, an invalid ``rank_constant``,
    a non-str ``document_id``, a rank that is not a positive int, or a
    ``document_id`` repeated within one list.
    """
    validate_top_k(top_k)
    validate_rank_constant(rank_constant)

    candidates: dict[str, _Candidate] = {}
    for results in ranked_lists:
        seen_in_source: set[str] = set()
        for result in results:
            _validate_result(result)
            if result.document_id in seen_in_source:
                raise ValueError(
                    f"duplicate document_id {result.document_id!r} in one ranked list"
                )
            seen_in_source.add(result.document_id)

            contribution = 1.0 / (rank_constant + result.rank)
            candidate = candidates.get(result.document_id)
            if candidate is None:
                candidates[result.document_id] = _Candidate(
                    contributions=[contribution],
                    best_rank=result.rank,
                    title=result.title,
                    snippet=result.snippet,
                )
                continue

            candidate.contributions.append(contribution)
            # Source lists are visited in order, so a strictly better rank is the
            # only thing that can replace the chosen metadata occurrence.
            if result.rank < candidate.best_rank:
                candidate.best_rank = result.rank
                candidate.title = result.title
                candidate.snippet = result.snippet

    ordered = sorted(
        candidates.items(),
        key=lambda item: (
            -math.fsum(item[1].contributions),
            item[1].best_rank,
            item[0],
        ),
    )

    return [
        SearchResult(
            document_id=document_id,
            rank=position,
            score=math.fsum(candidate.contributions),
            title=candidate.title,
            snippet=candidate.snippet,
        )
        for position, (document_id, candidate) in enumerate(ordered[:top_k], start=1)
    ]


def _validate_result(result: SearchResult) -> None:
    if not isinstance(result.document_id, str):
        raise ValueError("document_id must be a string")
    if (
        isinstance(result.rank, bool)
        or not isinstance(result.rank, int)
        or result.rank < 1
    ):
        raise ValueError(f"rank for {result.document_id!r} must be a positive integer")

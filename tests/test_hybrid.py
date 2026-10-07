"""Offline tests for HybridRetriever using fake Retriever implementations."""

import pytest

from fusion.hybrid import HybridRetriever
from fusion.rrf import reciprocal_rank_fusion
from retrievers.base import SearchResult


class FakeRetriever:
    """Returns canned results and records each search call."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        return self._results[:top_k]


def hit(doc_id, rank, score=0.0, title=None, snippet=None):
    return SearchResult(
        document_id=doc_id, rank=rank, score=score, title=title, snippet=snippet
    )


def test_forwards_same_query_and_requests_top_k_from_both():
    sparse, dense = FakeRetriever([]), FakeRetriever([])

    HybridRetriever(sparse, dense).search("질문", top_k=7)

    assert sparse.calls == [("질문", 7)]
    assert dense.calls == [("질문", 7)]


def test_combines_overlapping_results_with_rrf():
    sparse = FakeRetriever([hit("A", 1, 9.0), hit("B", 2, 8.0)])
    dense = FakeRetriever([hit("B", 1, 0.9), hit("C", 2, 0.8)])

    fused = HybridRetriever(sparse, dense).search("q", top_k=10)

    assert [r.document_id for r in fused] == ["B", "A", "C"]
    b = fused[0]
    assert b.rank == 1
    assert b.score == pytest.approx(1 / 61 + 1 / 62)


def test_result_matches_plain_rrf_over_child_outputs():
    sparse_results = [hit("A", 1), hit("B", 2)]
    dense_results = [hit("B", 1), hit("C", 2)]

    fused = HybridRetriever(
        FakeRetriever(sparse_results), FakeRetriever(dense_results), rank_constant=10
    ).search("q", top_k=5)

    expected = reciprocal_rank_fusion(
        [sparse_results, dense_results], top_k=5, rank_constant=10
    )
    assert fused == expected


def test_works_when_one_retriever_returns_nothing():
    dense = FakeRetriever([hit("C", 1)])

    fused = HybridRetriever(FakeRetriever([]), dense).search("q", top_k=3)

    assert [(r.document_id, r.rank) for r in fused] == [("C", 1)]


def test_works_when_both_retrievers_return_nothing():
    assert (
        HybridRetriever(FakeRetriever([]), FakeRetriever([])).search("q", top_k=3) == []
    )


def test_returns_at_most_top_k():
    sparse = FakeRetriever([hit(f"S{i}", i) for i in range(1, 6)])
    dense = FakeRetriever([hit(f"D{i}", i) for i in range(1, 6)])

    fused = HybridRetriever(sparse, dense).search("q", top_k=3)

    assert len(fused) == 3
    assert [r.rank for r in fused] == [1, 2, 3]


def test_score_is_rrf_score_not_child_raw_score():
    sparse = FakeRetriever([hit("A", 1, score=123.4)])
    dense = FakeRetriever([hit("A", 1, score=0.99)])

    (only,) = HybridRetriever(sparse, dense).search("q", top_k=1)

    assert only.score == pytest.approx(2 / 61)


def test_metadata_is_preserved_through_fusion():
    sparse = FakeRetriever([hit("A", 1, title="T", snippet="S")])
    dense = FakeRetriever([])

    (only,) = HybridRetriever(sparse, dense).search("q", top_k=1)

    assert (only.title, only.snippet) == ("T", "S")


def test_depends_only_on_search_protocol():
    class MinimalRetriever:
        def search(self, query, top_k):
            return [hit(query, 1)]

    fused = HybridRetriever(MinimalRetriever(), MinimalRetriever()).search(
        "doc-x", top_k=1
    )

    assert [r.document_id for r in fused] == ["doc-x"]


@pytest.mark.parametrize("bad_top_k", [0, -2, True])
def test_invalid_top_k_is_rejected_before_searching_children(bad_top_k):
    sparse, dense = FakeRetriever([]), FakeRetriever([])

    with pytest.raises(ValueError, match="top_k"):
        HybridRetriever(sparse, dense).search("q", top_k=bad_top_k)

    assert sparse.calls == [] and dense.calls == []


def test_invalid_rank_constant_is_rejected_at_construction():
    with pytest.raises(ValueError, match="rank_constant"):
        HybridRetriever(FakeRetriever([]), FakeRetriever([]), rank_constant=-1)

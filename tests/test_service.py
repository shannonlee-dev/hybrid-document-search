"""Offline tests for SearchService routing with fake retrievers."""

import pytest

from app.schemas import RetrievalMethod
from app.service import MethodState, MethodStatus, MethodUnavailableError, SearchService
from fusion.hybrid import HybridRetriever
from retrievers.base import SearchResult


class FakeRetriever:
    """Records calls and returns canned results, ignoring the query."""

    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return [
            SearchResult(
                document_id=doc_id, rank=rank, score=1.0 / rank, title=f"T{doc_id}"
            )
            for rank, doc_id in enumerate(self.ids[:top_k], start=1)
        ]


def test_each_method_routes_to_its_own_retriever():
    retrievers = {m: FakeRetriever([f"{m.value}-doc"]) for m in RetrievalMethod}
    service = SearchService(retrievers)

    for method in RetrievalMethod:
        results = service.search("q", method, 1)
        assert [r.document_id for r in results] == [f"{method.value}-doc"]

    # Each fake was called once, by its own method only.
    assert all(len(fake.calls) == 1 for fake in retrievers.values())


def test_query_and_top_k_are_forwarded_unchanged():
    fake = FakeRetriever(["a", "b", "c"])
    service = SearchService({RetrievalMethod.TFIDF: fake})

    service.search("대한민국의 수도", RetrievalMethod.TFIDF, 2)

    assert fake.calls == [("대한민국의 수도", 2)]


def test_ranking_and_metadata_are_preserved():
    service = SearchService({RetrievalMethod.DENSE: FakeRetriever(["x", "y"])})

    results = service.search("q", RetrievalMethod.DENSE, 5)

    assert [(r.document_id, r.rank) for r in results] == [("x", 1), ("y", 2)]
    assert [r.title for r in results] == ["Tx", "Ty"]


def test_results_are_truncated_to_top_k_and_keep_original_ranks():
    service = SearchService({RetrievalMethod.TFIDF: FakeRetriever(["a", "b", "c"])})

    results = service.search("q", RetrievalMethod.TFIDF, 2)

    assert [(r.document_id, r.rank) for r in results] == [("a", 1), ("b", 2)]


def test_empty_results_are_returned_as_empty_not_as_unavailable():
    service = SearchService({RetrievalMethod.TFIDF: FakeRetriever([])})

    assert service.search("q", RetrievalMethod.TFIDF, 3) == []


def test_unconfigured_method_raises_with_reason():
    service = SearchService(
        {RetrievalMethod.TFIDF: FakeRetriever(["a"])},
        statuses={
            RetrievalMethod.BM25: MethodStatus(
                MethodState.NOT_IMPLEMENTED, "BM25 retriever is not merged yet."
            )
        },
    )

    with pytest.raises(MethodUnavailableError) as excinfo:
        service.search("q", RetrievalMethod.BM25, 1)

    assert excinfo.value.method is RetrievalMethod.BM25
    assert excinfo.value.status.state is MethodState.NOT_IMPLEMENTED
    assert "not merged yet" in str(excinfo.value)


def test_unconfigured_method_without_reason_says_not_configured():
    service = SearchService({})

    with pytest.raises(MethodUnavailableError, match="not configured"):
        service.search("q", RetrievalMethod.DENSE, 1)


def test_availability_reports_every_method():
    service = SearchService(
        {RetrievalMethod.TFIDF: FakeRetriever(["a"])},
        statuses={
            RetrievalMethod.BM25: MethodStatus(MethodState.INDEX_MISSING, "missing")
        },
    )

    availability = service.availability()

    assert list(availability) == list(RetrievalMethod)
    assert availability[RetrievalMethod.TFIDF].available
    assert availability[RetrievalMethod.BM25].state is MethodState.INDEX_MISSING
    assert availability[RetrievalMethod.DENSE].state is MethodState.NOT_CONFIGURED


def test_hybrid_routing_fuses_sparse_and_dense_with_rrf():
    sparse = FakeRetriever(["a", "b", "c"])
    dense = FakeRetriever(["b", "d", "a"])
    hybrid = HybridRetriever(sparse, dense)
    service = SearchService({RetrievalMethod.HYBRID: hybrid})

    results = service.search("q", RetrievalMethod.HYBRID, 4)

    # b: 1/62 + 1/61 beats a: 1/61 + 1/63, so b is ranked first.
    assert [r.document_id for r in results] == ["b", "a", "d", "c"]
    assert [r.rank for r in results] == [1, 2, 3, 4]
    assert sparse.calls == [("q", 4)]
    assert dense.calls == [("q", 4)]


def test_repeated_identical_searches_are_deterministic():
    service = SearchService({RetrievalMethod.TFIDF: FakeRetriever(["a", "b"])})

    first = service.search("q", RetrievalMethod.TFIDF, 2)
    second = service.search("q", RetrievalMethod.TFIDF, 2)

    assert first == second

"""질의 순서, 임의 Top-K 및 검색 호출만의 시간 측정을 검증합니다."""

import pytest

from retrievers.base import SearchResult


def test_measurement_preserves_duplicate_queries_top_k_and_first_results(monkeypatch):
    from evaluation import latency

    calls = []
    elapsed = 0

    class Retriever:
        def search(self, query, top_k):
            nonlocal elapsed
            calls.append((query, top_k))
            elapsed += len(calls)
            return [SearchResult(str(len(calls)), 1, 1.0)]

    def _validate(position, results):
        nonlocal elapsed
        assert position in (0, 1, 2)
        assert len(results) == 1
        # Validation must not contribute to measured request latency.
        elapsed += 100

    monkeypatch.setattr(latency, "perf_counter", lambda: elapsed)
    result = latency.measure_query_latency(
        Retriever(),
        ["cat", "dog", "cat"],
        top_k=20,
        warmup=1,
        repeats=2,
        validate_results=_validate,
    )
    assert calls == [("cat", 20), ("dog", 20), ("cat", 20)] * 3
    assert result.warmup_seconds == 306
    assert result.samples_ms == [[4000, 7000], [5000, 8000], [6000, 9000]]
    assert [hits[0].document_id for hits in result.first_results] == ["4", "5", "6"]
    assert result.latency == pytest.approx(
        {"mean_ms": 6500, "p95_ms": 8750, "sample_count": 6}
    )

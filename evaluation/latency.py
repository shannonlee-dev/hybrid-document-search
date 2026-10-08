"""품질 평가와 독립적으로 검색 지연을 측정하고 밀리초 단위로 집계합니다."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import ceil, fsum, isfinite
from numbers import Real
from time import perf_counter

from evaluation.metrics import EVALUATION_TOP_K
from retrievers.base import Retriever, SearchResult

DEFAULT_WARMUP = 1
DEFAULT_REPEATS = 5
MILLISECONDS_PER_SECOND = 1000
_P95_QUANTILE = 0.95


@dataclass(frozen=True)
class LatencyMeasurement:
    """입력 위치별 시간 샘플과 품질 평가에서 재사용할 첫 측정 검색 결과입니다."""

    latency: dict[str, float | int]
    warmup_seconds: float
    samples_ms: list[list[float]]
    first_results: list[list[SearchResult]]


def latency_summary(samples: list[float]) -> dict[str, float | int]:
    """전체 요청 샘플의 평균과 선형 보간 P95를 계산합니다(단위 ms)."""
    if not samples or any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
        or value < 0
        for value in samples
    ):
        raise ValueError(
            "latency 샘플은 비어 있지 않은 유한한 음이 아닌 숫자 목록이어야 합니다."
        )
    ordered = sorted(samples)
    position = (len(ordered) - 1) * _P95_QUANTILE
    lower = int(position)
    upper = ceil(position)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {
        "mean_ms": fsum(samples) / len(samples),
        "p95_ms": p95,
        "sample_count": len(samples),
    }


def measure_query_latency(
    retriever: Retriever,
    queries: Sequence[str],
    *,
    top_k: int = EVALUATION_TOP_K,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
    validate_results: Callable[[int, list[SearchResult]], None] | None = None,
) -> LatencyMeasurement:
    """고정 순서로 예열·반복하며 search 호출만 측정합니다.

    같은 텍스트의 질의도 입력 위치별로 별도 요청으로 보존합니다. 선택적인
    결과 검증은 예열 및 매 측정 검색 직후, 요청 타이머 밖에서 수행합니다.
    예열 전체 시간에는 결과 검증이 포함되며 품질 지표는 계산하지 않습니다.
    """
    for name, value in (("top_k", top_k), ("warmup", warmup), ("repeats", repeats)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name}는 양의 정수여야 합니다.")
    if (
        isinstance(queries, (str, bytes))
        or not isinstance(queries, Sequence)
        or not queries
        or any(not isinstance(query, str) or not query.strip() for query in queries)
    ):
        raise ValueError("queries는 비어 있지 않은 질의 문자열 목록이어야 합니다.")

    started = perf_counter()
    for _ in range(warmup):
        for position, query in enumerate(queries):
            results = retriever.search(query, top_k)
            if validate_results is not None:
                validate_results(position, results)
    warmup_seconds = perf_counter() - started

    samples = [[] for _ in queries]
    first_results = []
    for repetition in range(repeats):
        for position, query in enumerate(queries):
            started = perf_counter()
            results = retriever.search(query, top_k)
            elapsed_ms = (perf_counter() - started) * MILLISECONDS_PER_SECOND
            if validate_results is not None:
                validate_results(position, results)
            samples[position].append(elapsed_ms)
            if repetition == 0:
                first_results.append(results)
    return LatencyMeasurement(
        latency=latency_summary([sample for values in samples for sample in values]),
        warmup_seconds=warmup_seconds,
        samples_ms=samples,
        first_results=first_results,
    )

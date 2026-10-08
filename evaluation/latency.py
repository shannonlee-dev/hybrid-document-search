"""Measure search latency in milliseconds independently of quality evaluation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import ceil, fsum, isfinite
from numbers import Real
from time import perf_counter

from evaluation.constants import DEFAULT_REPEATS as DEFAULT_REPEATS
from evaluation.constants import DEFAULT_WARMUP as DEFAULT_WARMUP
from evaluation.constants import EVALUATION_TOP_K
from evaluation.constants import MILLISECONDS_PER_SECOND as MILLISECONDS_PER_SECOND
from retrievers.base import Retriever, SearchResult

_P95_QUANTILE = 0.95


@dataclass(frozen=True)
class LatencyMeasurement:
    """Per-query latency samples and first measured results for quality evaluation."""

    latency: dict[str, float | int]
    warmup_seconds: float
    samples_ms: list[list[float]]
    first_results: list[list[SearchResult]]


def latency_summary(samples: list[float]) -> dict[str, float | int]:
    """Compute mean latency and linearly interpolated P95 from millisecond samples."""
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
    """Warm up and repeat queries in a fixed order, timing only search calls.

    Measure duplicate queries by input position. Validate results after each search;
    request latency excludes validation, while total warm-up time includes it.
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

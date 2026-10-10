"""Compute Recall, RR/MRR and nDCG from shared search rankings and binary qrels."""

from collections.abc import Iterable, Mapping
from math import fsum, isfinite, log2
from numbers import Real

from evaluation.constants import EVALUATION_TOP_K as EVALUATION_TOP_K
from evaluation.constants import METRIC_NAMES as METRIC_NAMES
from retrievers.base import SearchResult


def recall_at_k(
    results: Iterable[SearchResult], qrels: Mapping[str, float], k: int
) -> float:
    """Return the fraction of positively judged documents found in the top k results."""
    document_ids, relevant = _validate_inputs(results, qrels, k)
    return _recall(document_ids, relevant, k)


def reciprocal_rank_at_k(
    results: Iterable[SearchResult], qrels: Mapping[str, float], k: int
) -> float:
    """Return the reciprocal rank of the first relevant top-k result, or zero."""
    document_ids, relevant = _validate_inputs(results, qrels, k)
    return _reciprocal_rank(document_ids, relevant, k)


def ndcg_at_k(
    results: Iterable[SearchResult], qrels: Mapping[str, float], k: int
) -> float:
    """Compute binary nDCG@K, treating positive relevance as a gain of one."""
    document_ids, relevant = _validate_inputs(results, qrels, k)
    return _ndcg(document_ids, relevant, k)


def evaluate_query(
    results: Iterable[SearchResult], qrels: Mapping[str, float]
) -> dict[str, float]:
    """Compute Recall@5/10, RR@10 and nDCG@10 for one query."""
    document_ids, relevant = _validate_inputs(results, qrels, EVALUATION_TOP_K)
    return {
        "recall@5": _recall(document_ids, relevant, 5),
        "recall@10": _recall(document_ids, relevant, EVALUATION_TOP_K),
        "rr@10": _reciprocal_rank(document_ids, relevant, EVALUATION_TOP_K),
        "ndcg@10": _ndcg(document_ids, relevant, EVALUATION_TOP_K),
    }


def evaluate_run(
    run: Mapping[str, Iterable[SearchResult]],
    qrels: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Compute macro-averaged metrics across all judged queries.

    Missing queries use empty results. Reject results for unknown query IDs and
    empty qrels, for which the mean is undefined.
    """
    if not isinstance(run, Mapping) or not isinstance(qrels, Mapping):
        raise TypeError("run과 qrels는 query_id를 키로 갖는 mapping이어야 합니다.")
    if not qrels:
        raise ValueError("평가할 질의가 하나 이상 필요합니다.")
    for query_id in (*run, *qrels):
        _validate_id(query_id, "query_id")
    if set(run) - set(qrels):
        raise ValueError("qrels에 없는 query_id의 검색 결과가 있습니다.")

    per_query = [
        evaluate_query(run.get(query_id, ()), judgments)
        for query_id, judgments in qrels.items()
    ]
    names = ("recall@5", "recall@10", "rr@10", "ndcg@10")
    averages = {}
    # 질의별 RR의 평균은 보고서에서 MRR로 표기한다.
    for name, query_name in zip(METRIC_NAMES, names, strict=True):
        values = [metrics[query_name] for metrics in per_query]
        averages[name] = fsum(values) / len(per_query)
    return averages


def _validate_inputs(
    results: Iterable[SearchResult], qrels: Mapping[str, float], k: int
) -> tuple[list[str], set[str]]:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k는 양의 정수여야 합니다.")
    if not isinstance(qrels, Mapping):
        raise TypeError("qrels는 document_id와 relevance를 담은 mapping이어야 합니다.")
    relevant = set()
    for document_id, relevance in qrels.items():
        _validate_id(document_id, "document_id")
        if (
            isinstance(relevance, bool)
            or not isinstance(relevance, Real)
            or not isfinite(relevance)
        ):
            raise ValueError("relevance는 bool을 제외한 유한한 숫자여야 합니다.")
        if relevance > 0:
            relevant.add(document_id)

    document_ids = []
    seen = set()
    for rank, result in enumerate(results, start=1):
        if not isinstance(result, SearchResult):
            raise TypeError("results의 각 항목은 SearchResult여야 합니다.")
        _validate_id(result.document_id, "document_id")
        if result.document_id in seen:
            raise ValueError("검색 결과에 중복 document_id가 있습니다.")
        if (
            isinstance(result.rank, bool)
            or not isinstance(result.rank, int)
            or result.rank != rank
        ):
            raise ValueError("검색 결과 rank는 목록 순서대로 1부터 연속해야 합니다.")
        seen.add(result.document_id)
        document_ids.append(result.document_id)
    return document_ids, relevant


def _validate_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}는 비어 있지 않은 문자열이어야 합니다.")


def _recall(document_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(document_ids[:k]) & relevant) / len(relevant)


def _reciprocal_rank(document_ids: list[str], relevant: set[str], k: int) -> float:
    for rank, document_id in enumerate(document_ids[:k], start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg(document_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = fsum(
        1.0 / log2(rank + 1)
        for rank, document_id in enumerate(document_ids[:k], start=1)
        if document_id in relevant
    )
    ideal_dcg = fsum(
        1.0 / log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1)
    )
    return dcg / ideal_dcg

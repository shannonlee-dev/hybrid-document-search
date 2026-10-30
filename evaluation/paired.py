"""Query-level paired comparison of two retrieval runs on the same query set.

The bootstrap resamples query indices with replacement and applies the same
indices to both runs, so the interval is for the mean paired difference
(B - A). It is a descriptive interval, not a multiple-comparison corrected test.
"""

from math import fsum

PAIRED_METRICS = ("ndcg@10", "mrr@10", "recall@10", "p@1")
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20261030
_TIE_TOLERANCE = 1e-12


def per_query_metrics(report, method):
    """Map query ID to the metric dict of one method in an enterprise report."""
    return {
        row["query_id"]: row["metrics"] for row in report["methods"][method]["queries"]
    }


def paired_comparison(
    a,
    b,
    query_ids,
    *,
    metrics=PAIRED_METRICS,
    resamples=DEFAULT_RESAMPLES,
    seed=DEFAULT_SEED,
):
    """Compare per-query metric dicts `a` and `b` on `query_ids`."""
    import numpy as np

    query_ids = list(query_ids)
    if not query_ids or len(set(query_ids)) != len(query_ids):
        raise ValueError("query_ids는 중복 없는 비어 있지 않은 목록이어야 합니다.")
    missing = [q for q in query_ids if q not in a or q not in b]
    if missing:
        raise ValueError(f"두 run 중 하나에 없는 질의 {len(missing)}개")
    if type(resamples) is not int or resamples < 1:
        raise ValueError("resamples는 양의 정수여야 합니다.")
    count = len(query_ids)
    indices = np.random.default_rng(seed).integers(0, count, size=(resamples, count))
    result = {
        "query_count": count,
        "resamples": resamples,
        "seed": seed,
        "interval": "percentile 2.5/97.5 of resampled mean(B - A)",
        "tie_tolerance": _TIE_TOLERANCE,
        "metrics": {},
    }
    for metric in metrics:
        left = np.array([a[q][metric] for q in query_ids], dtype=np.float64)
        right = np.array([b[q][metric] for q in query_ids], dtype=np.float64)
        delta = right - left
        boot = delta[indices].mean(axis=1)
        low, high = np.percentile(boot, [2.5, 97.5])
        result["metrics"][metric] = {
            "mean_a": fsum(left) / count,
            "mean_b": fsum(right) / count,
            "difference": fsum(delta) / count,
            "ci95": [float(low), float(high)],
            "ci_excludes_zero": bool(low > 0 or high < 0),
            "wins_b": int((delta > _TIE_TOLERANCE).sum()),
            "ties": int((np.abs(delta) <= _TIE_TOLERANCE).sum()),
            "losses_b": int((delta < -_TIE_TOLERANCE).sum()),
        }
    return result

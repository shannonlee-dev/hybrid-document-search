"""Query 단위 paired difference와 bootstrap 구간을 고정 입력으로 검증합니다."""

import copy

import pytest

pytest.importorskip("numpy")

from evaluation.paired import paired_comparison  # noqa: E402


def _metrics(values):
    return {
        q: {"ndcg@10": v, "mrr@10": v, "recall@10": v, "p@1": v}
        for q, v in values.items()
    }


def test_paired_comparison_counts_and_interval():
    a = _metrics({"q1": 0.0, "q2": 1.0, "q3": 0.5, "q4": 0.5})
    b = _metrics({"q1": 1.0, "q2": 1.0, "q3": 0.0, "q4": 1.0})
    result = paired_comparison(a, b, ["q1", "q2", "q3", "q4"], resamples=500, seed=1)
    ndcg = result["metrics"]["ndcg@10"]
    assert ndcg["difference"] == pytest.approx(0.25)
    assert (ndcg["wins_b"], ndcg["ties"], ndcg["losses_b"]) == (2, 1, 1)
    assert ndcg["ci95"][0] <= ndcg["difference"] <= ndcg["ci95"][1]
    assert result["resamples"] == 500 and result["seed"] == 1
    again = paired_comparison(a, b, ["q1", "q2", "q3", "q4"], resamples=500, seed=1)
    assert again == result


def test_paired_comparison_identical_runs_have_zero_interval():
    a = _metrics({"q1": 0.3, "q2": 0.7})
    result = paired_comparison(a, copy.deepcopy(a), ["q1", "q2"], resamples=50)
    ndcg = result["metrics"]["ndcg@10"]
    assert ndcg["ci95"] == [0.0, 0.0]
    assert not ndcg["ci_excludes_zero"]
    assert ndcg["ties"] == 2


def test_paired_comparison_rejects_missing_queries():
    a = _metrics({"q1": 0.3})
    with pytest.raises(ValueError, match="없는 질의"):
        paired_comparison(a, a, ["q1", "q2"])

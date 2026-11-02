"""The committed RAGAS summary stays internally consistent with the recorded run."""

import json
from pathlib import Path

import pytest

from evaluation.rag_downstream import (
    MAX_QUERIES,
    PILOT_QUERIES,
    RAG_CONTEXT_TOP_K,
    RAG_METHODS,
    RAG_SEED,
)
from scripts.evaluate_rag_downstream import GENERATION_PARAMS

SUMMARY = (
    Path(__file__).resolve().parents[1]
    / "docs/experiments/ragas/summary-strategyqa-100.json"
)


@pytest.fixture(scope="module")
def summary():
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_run_conditions_match_the_code(summary):
    subset = summary["config"]["subset"]
    assert summary["queries"] == subset["selected"] == MAX_QUERIES == 100
    assert subset["seed"] == RAG_SEED == 20261023
    assert subset["pilot"] == PILOT_QUERIES == 50
    assert summary["generation"]["params"] == GENERATION_PARAMS
    assert summary["generation"]["params"]["temperature"] == 0.0
    assert summary["judge"]["params"]["temperature"] == 0.0
    assert summary["judge"]["ragas_version"] == "0.4.3"
    assert RAG_CONTEXT_TOP_K == 5
    assert set(summary["methods"]) == set(RAG_METHODS)


def test_failures_are_recorded_not_dropped(summary):
    truncated = {(t["method"], t["query_id"]) for t in summary["truncated_generations"]}
    assert truncated == {
        ("tfidf", "0b7787655f4a9d600dfc_0"),
        ("rrf", "01254b50648f4fb08b51_1"),
    }
    judge_errors = {
        (method, metric): count
        for method, table in summary["methods"].items()
        for metric, count in table["judge_errors"].items()
        if count
    }
    assert judge_errors == {
        ("score_fusion", "context_precision"): 1,
        ("score_fusion", "faithfulness"): 1,
    }
    for table in summary["methods"].values():
        assert table["generation_errors"] == 0
        for metric in ("recall@5", "rr@10", "ndcg@10"):
            assert table[metric]["n"] == 100 and table[metric]["undefined"] == 0


@pytest.mark.parametrize(
    "metric, delta, low, high, n",
    [
        ("ndcg@10", 0.0060, -0.0110, 0.0225, 100),
        ("id_context_recall", 0.0208, -0.0008, 0.0488, 100),
        ("context_precision", -0.0078, -0.0341, 0.0180, 99),
        ("faithfulness", -0.0101, -0.0448, 0.0221, 99),
        ("response_relevancy", -0.0092, -0.0349, 0.0099, 100),
    ],
)
def test_reported_score_fusion_minus_dense_values(summary, metric, delta, low, high, n):
    result = summary["paired"]["score_fusion-dense"]["metrics"][metric]
    assert result["mean_delta"] == pytest.approx(delta, abs=5e-5)
    assert result["ci"] == pytest.approx([low, high], abs=5e-5)
    assert result["n"] == n
    assert result["ci"][0] <= result["mean_delta"] <= result["ci"][1]


def test_cluster_bootstrap_groups_and_no_query_text(summary):
    for comparison in summary["paired"].values():
        for result in comparison["metrics"].values():
            assert result["groups"] <= result["n"]
        assert "differing_queries" not in comparison
    assert summary["paired"]["score_fusion-dense"]["metrics"]["ndcg@10"]["groups"] == 96
    assert "prompt_template" not in summary["config"]

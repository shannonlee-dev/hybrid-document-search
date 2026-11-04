"""Check the committed external baseline review evidence (Finance vs Kanana)."""

import json

import pytest

pytest.importorskip("numpy")

from scripts import analyze_external_baseline as ext  # noqa: E402


def _load(name):
    return json.loads((ext.EVIDENCE / name).read_text(encoding="utf-8"))


def test_comparison_recomputes_from_committed_rows():
    assert ext.check() == {"verified": True}


def test_pipelines_agree_before_cross_model_comparison():
    gate = _load("comparison.json")["gate"]
    assert gate["passed"]
    assert set(gate["phase5_vs_service_rank_mismatches"].values()) == {0}


def test_reference_rows_reproduce_published_intervals():
    replay = _load("comparison.json")["reference_replay"]
    phase5 = replay["phase5_bge_kanana_dense_seed20261030_10000"]
    assert round(phase5["difference"], 4) == 0.0194
    assert [round(x, 4) for x in phase5["ci95"]] == [0.0058, 0.0330]
    pr42 = replay["pr42_bge_fin_fixed_seed20261102_20000"]
    assert [round(x, 4) for x in pr42["ci95"]] == [0.0026, 0.0328]


def test_metrics_from_rank():
    assert ext.metrics_from_rank(None) == dict.fromkeys(ext.METRICS, 0.0)
    m = ext.metrics_from_rank(3)
    assert m["ndcg@10"] == 0.5 and m["mrr@10"] == 1 / 3
    assert m["recall@5"] == m["recall@10"] == 1.0 and m["p@1"] == 0.0


def test_leakage_ids_are_primary_queries_above_threshold():
    leak = _load("leakage.json")["query_side"]
    sims = leak["max_similarity_by_query"]
    near = [q for q, s in sims.items() if s >= ext.NEAR_DUP]
    assert sorted(near) == sorted(leak["near_duplicate_query_ids"])
    assert len(near) == leak["bands"][">=0.90"]

"""Check the committed Finance encoder service-path evidence."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/validation/2026-11-02/finance-service-eval"


def _load(name):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def _rows(name):
    return [json.loads(line) for line in (EVIDENCE / name).read_text().splitlines()]


def test_manifest_analysis_and_figures_recompute():
    spec = importlib.util.spec_from_file_location(
        "analyze_finance_service", ROOT / "scripts/analyze_finance_service.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.check(EVIDENCE) == {"verified": True, "files": 23}


def test_parity_checks_passed_before_accuracy():
    parity = _load("parity.json")
    assert parity["passed"]
    assert parity["check1_checkpoint"]["bitwise_equal"]
    accuracy = _load("accuracy.json")
    assert set(accuracy["api_vs_direct_top10_mismatches"].values()) == {0}


def test_preregistered_bccard_verdicts():
    comparisons = _load("accuracy.json")["comparisons"]
    assert comparisons["A"]["ndcg@10"]["verdict"] == "maintained"
    assert comparisons["B"]["ndcg@10"]["verdict"] == "maintained"
    assert comparisons["A"]["ndcg@10"]["ci95"][0] > 0


def test_external_verdicts_are_reported_as_measured():
    external = _load("external.json")
    for pair in ("A", "B", "D"):
        assert external["strategyqa"]["comparisons"][pair]["ndcg@10"]["verdict"] == "reversed"
        assert external["miracl10k"]["comparisons"][pair]["ndcg@10"]["verdict"] == "not_shown"
    for name in ("miracl10k", "strategyqa"):
        for check in external[name]["path_check_cached_bge"].values():
            assert check["max_abs_difference"] == 0


def test_per_query_rows_have_ids_only():
    rows = _rows("bccard-per-query.jsonl")
    assert len(rows) == 7 * 721
    assert len({(r["condition"], r["query_id"]) for r in rows}) == len(rows)
    assert not any("text" in r or "query" in r for r in rows)
    assert len(_rows("strategyqa-per-query.jsonl")) == 6 * 592
    assert len(_rows("miracl10k-per-query.jsonl")) == 6 * 50

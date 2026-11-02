"""Checks for the exploratory Ko-StrategyQA regression evidence and helpers."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/validation/2026-11-02/finance-strategyqa-analysis"


def _module():
    spec = importlib.util.spec_from_file_location(
        "analyze_strategyqa_regression",
        ROOT / "scripts/analyze_strategyqa_regression.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary():
    return [
        json.loads(line)
        for line in (EVIDENCE / "per-query-summary.jsonl").read_text().splitlines()
    ]


def test_manifest_and_analysis_recompute():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    assert _module().check(EVIDENCE) == {"verified": True, "files": 8}


def test_summary_has_ids_and_numbers_only():
    rows = _summary()
    assert sum(r["dataset"] == "strategyqa" for r in rows) == 592
    assert sum(r["dataset"] == "bccard" for r in rows) == 721
    allowed = (str, int, float, bool)
    for row in rows:
        assert not {"text", "query", "title"} & set(row)
        assert all(isinstance(v, allowed) for v in row.values())


def test_manual_review_ids_exist_and_posthoc_is_marked():
    ids = {r["query_id"] for r in _summary() if r["dataset"] == "strategyqa"}
    review = json.loads((EVIDENCE / "manual-review.json").read_text())
    assert {c["query_id"] for c in review["cases"]} <= ids
    assert len(review["cases"]) == 20
    posthoc = json.loads((EVIDENCE / "posthoc-same-source.json").read_text())
    assert posthoc["kind"].startswith("post-hoc")


def test_lexical_overlap_and_movement_helpers():
    module = _module()
    assert module.lexical_overlap("카드 해지", ["카드해지"]) == 1.0
    assert module.lexical_overlap("가나", ["다라"]) == 0.0
    rows = [
        {"first_relevant_rank_bge": 12, "first_relevant_rank_fin": 3},
        {"first_relevant_rank_bge": 2, "first_relevant_rank_fin": 11},
        {"first_relevant_rank_bge": 4, "first_relevant_rank_fin": 1},
        {"first_relevant_rank_bge": 1, "first_relevant_rank_fin": 1},
        {"first_relevant_rank_bge": 50, "first_relevant_rank_fin": 70},
    ]
    assert module.movement(rows) == {
        "entered_top10": 1,
        "left_top10": 1,
        "up_in_top10": 1,
        "down_in_top10": 0,
        "same_in_top10": 1,
        "outside_both": 1,
    }

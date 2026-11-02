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
    assert _module().check(EVIDENCE) == {"verified": True, "files": 14}


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


def test_faiss_and_numpy_order_exact_ties_differently():
    """Minimal reproduction of the v1 parity gap: identical vectors tie exactly."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("faiss")
    from indexing.faiss_index import FaissIndex

    vectors = np.array([[0.6, 0.8], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    faiss_rows = [row for row, _ in FaissIndex.build(vectors).search(query, 3)]
    numpy_rows = np.argsort(-(vectors @ query[0]), kind="stable").tolist()
    assert faiss_rows[2] == 0 and numpy_rows[2] == 0
    assert set(faiss_rows[:2]) == set(numpy_rows[:2]) == {1, 2}
    # FAISS put the higher corpus row first for every observed tie (79/79 cases).
    assert faiss_rows[:2] == [2, 1] and numpy_rows[:2] == [1, 2]


def test_rank_features_match_canonical_service_rows():
    """v2 rank features must agree with the committed canonical Dense rows."""
    service = ROOT / "docs/validation/2026-11-02/finance-service-eval"
    canonical = {}
    for name in ("strategyqa-per-query.jsonl", "bccard-per-query.jsonl"):
        for line in (service / name).read_text().splitlines():
            row = json.loads(line)
            canonical[(row["condition"], row["query_id"])] = row
    checked = 0
    for row in _summary():
        for encoder in ("bge", "fin"):
            committed = canonical[(f"D-{encoder}", row["query_id"])]
            rank = row[f"first_relevant_rank_{encoder}"]
            if committed["relevant_rank_top10"] is None:
                assert rank > 10
            else:
                assert rank == committed["relevant_rank_top10"]
                checked += 1
    assert checked > 1000


def test_ranking_parity_attributes_every_mismatch_to_exact_ties():
    parity = json.loads((EVIDENCE / "ranking-parity.json").read_text())
    for dataset in parity["datasets"].values():
        for totals in dataset["encoders"].values():
            assert totals["faiss_now_equals_canonical"] == totals["queries"]
            assert totals["mismatch_other"] == 0
            assert totals["query_vector_reencode_bitwise_equal"] == totals["queries"]
    features = json.loads((EVIDENCE / "features-parity.json").read_text())
    for dataset in features["datasets"].values():
        assert dataset["dense_top10_mismatches"] == {"bge": 0, "fin": 0}

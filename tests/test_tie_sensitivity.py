"""Tie-order sensitivity of the PR #42 BC Card metrics (post-hoc) and v1 provenance."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/validation/2026-11-02/finance-strategyqa-analysis"
SERVICE = ROOT / "docs/validation/2026-11-02/finance-service-eval"


def _module():
    spec = importlib.util.spec_from_file_location(
        "analyze_tie_sensitivity", ROOT / "scripts/analyze_tie_sensitivity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows():
    return [
        json.loads(line)
        for line in (EVIDENCE / "tie-sensitivity-per-query.jsonl")
        .read_text()
        .splitlines()
    ]


def test_tie_rules_order_equal_scores_by_row():
    m = _module()
    scores, rows = [0.9, 0.5, 0.5, 0.5, 0.1], [7, 3, 9, 1, 4]
    assert m.order_by(scores, rows, "row_ascending") == [7, 1, 3, 9, 4]
    assert m.order_by(scores, rows, "row_descending") == [7, 9, 3, 1, 4]
    assert m.tie_blocks([0.9, 0.5, 0.5, 0.5, 0.1]) == [(0, 1), (1, 4), (4, 5)]


def test_bounds_handle_a_tie_block_crossing_the_cutoff():
    m = _module()
    # Positions 2-4 hold one exact tie; k=3 cuts through it.
    rows, scores = [10, 11, 12, 13], [0.9, 0.5, 0.5, 0.5]
    relevant = {13}
    assert m.bound_order(rows, scores, relevant, best=True, k=3) == [10, 13, 11]
    worst = m.bound_order(rows, scores, relevant, best=False, k=3)
    assert worst == [10, 11, 12] and 13 not in worst


def test_every_ordering_lies_within_the_tie_bounds():
    for row in _rows():
        for metric in ("ndcg@10", "mrr@10", "recall@10", "p@1"):
            low, high = row["lower_bound"][metric], row["upper_bound"][metric]
            for ordering in ("canonical", "row_ascending", "row_descending"):
                assert low - 1e-12 <= row[ordering][metric] <= high + 1e-12


def test_metric_changes_only_where_a_tie_block_mixes_labels():
    for row in _rows():
        values = {
            row[o]["ndcg@10"]
            for o in ("canonical", "row_ascending", "lower_bound", "upper_bound")
        }
        if len(values) > 1:
            assert row["top10_tie_block_label_conflict"]
            assert row["tie_block_has_duplicate_vectors"]
    audit = json.loads((EVIDENCE / "duplicate-label-audit.json").read_text())
    assert audit["duplicate_vector_groups"] == 192
    assert audit["documents_in_duplicate_groups"] == 392
    assert audit["queries_with_binary_label_conflict"] == 10
    summary = json.loads((EVIDENCE / "tie-sensitivity.json").read_text())
    for condition in summary["conditions"].values():
        for count in condition["tie_sensitive_queries"].values():
            assert count <= audit["queries_with_binary_label_conflict"]


def test_canonical_rankings_reproduce_the_committed_pr42_rows():
    audit = json.loads((EVIDENCE / "duplicate-label-audit.json").read_text())
    assert set(audit["canonical_reproduction_matches_committed_top10"].values()) == {
        721
    }
    source = SERVICE / "bccard-per-query.jsonl"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert audit["source_rows_sha256"] == digest
    assert (
        f"{digest}  bccard-per-query.jsonl" in (SERVICE / "MANIFEST.sha256").read_text()
    )


def test_sensitivity_summary_recomputes():
    pytest.importorskip("numpy")
    assert _module().check(EVIDENCE) == {"verified": True}


def test_removed_v1_files_are_recoverable_from_git_history():
    revisions = json.loads((EVIDENCE / "revisions.json").read_text())
    commit = revisions["v1_commit"]
    present = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT, capture_output=True
    )
    if present.returncode != 0:
        pytest.skip("git history for the v1 commit is not available (shallow clone)")
    for name, entry in revisions["files"].items():
        assert not (
            EVIDENCE / entry["path_at_renamed_copy_commit"].rsplit("/", 1)[1]
        ).exists()
        for ref, path in (
            (commit, entry["path_at_v1_commit"]),
            (revisions["renamed_copy_commit"], entry["path_at_renamed_copy_commit"]),
        ):
            blob = subprocess.run(
                ["git", "show", f"{ref}:{path}"],
                cwd=ROOT,
                capture_output=True,
                check=True,
            ).stdout
            assert hashlib.sha256(blob).hexdigest() == entry["sha256"], name

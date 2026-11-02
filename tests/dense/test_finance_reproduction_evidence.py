"""Check the committed Blackwell two-seed reproduction evidence."""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/validation/2026-11-02/finance-repro-blackwell"


def _analysis_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "analyze_finance_reproduction",
        ROOT / "scripts/analyze_finance_reproduction.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_analysis_and_figures_recompute(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert _analysis_module().check(EVIDENCE) == {"verified": True, "files": 12}


def test_loss_log_launcher_is_the_recorded_original():
    import hashlib

    launcher = ROOT / "scripts/train_query_encoder_with_loss_log.py"
    environment = json.loads((EVIDENCE / "environment.json").read_text())
    digest = hashlib.sha256(launcher.read_bytes()).hexdigest()
    assert digest == environment["launcher"]["sha256"]
    assert digest == (
        "1d632786ea9aec15812ff92329aef1f551a37afbc3fcc7f9edc80ee5f14d2777"
    )


def test_frozen_selection_and_policies_are_recorded_as_run():
    frozen = json.loads((EVIDENCE / "frozen.json").read_text())
    assert frozen["selected_encoder"] == "seed43"
    assert frozen["shared_alpha_dense"] == 0.84
    alphas = {k: round(v["alpha_dense"], 2) for k, v in frozen["models"].items()}
    assert alphas == {"baseline": 0.84, "seed42": 0.88, "seed43": 0.89}
    protocol = json.loads((EVIDENCE / "protocol.json").read_text())
    steps = {k: v["selected_step"] for k, v in protocol["encoders"].items()}
    assert steps == {"seed42": 2048, "seed43": 4096}
    for encoder in protocol["encoders"].values():
        training = encoder["training"]
        assert training["steps"] == 4096 and training["batch"] == 16
        assert (
            training["code"]["git_commit"] == "b6f8649a8dc7a8cbb580926ba2cc08cbee9358df"
        )
        assert training["code"]["working_tree_dirty"] is False
        assert (
            training["training_data_audit"]["excluded_exact_duplicate_examples"] == 244
        )
    assert protocol["vector_sha256"]["document-vectors.npy"] == (
        "4d5f1c8d9002bfeac0c7a927aa641b0ab17199ea008ea196fab786b08bb3df3a"
    )


def test_per_query_rows_cover_three_encoders_two_policies():
    rows = [
        json.loads(line)
        for line in (EVIDENCE / "per-query.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4326
    assert len({(r["method"], r["query_id"]) for r in rows}) == 4326
    assert not any("text" in r for r in rows)


def test_bootstrap_is_post_hoc_and_seeded():
    analysis = json.loads((EVIDENCE / "analysis.json").read_text())
    bootstrap = analysis["bootstrap"]
    assert bootstrap["seed"] == 20261102 and bootstrap["resamples"] >= 10000
    assert "not a new held-out experiment" in bootstrap["kind"]
    low, high = bootstrap["primary"]["ci95_percentile"]
    assert 0 < low < bootstrap["primary"]["mean_difference"] < high


@pytest.mark.skipif(
    not os.getenv("FINANCE_DATA_DIR"), reason="official local data not configured"
)
def test_reproduction_verifies_against_official_data():
    from evaluation.query_encoder_report import verify_results

    result = verify_results(Path(os.environ["FINANCE_DATA_DIR"]), EVIDENCE)
    assert result["queries"] == 721 and result["rows"] == 4326
    aggregates = result["aggregates"]
    assert aggregates["baseline + Fixed alpha=0.84"]["ndcg@10"] == pytest.approx(
        0.7912423386867907, abs=1e-12
    )
    assert aggregates["seed43 + Fixed alpha=0.84"]["ndcg@10"] == pytest.approx(
        0.8123864441432933, abs=1e-12
    )

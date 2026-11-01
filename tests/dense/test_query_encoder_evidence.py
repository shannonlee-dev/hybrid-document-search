"""Verify the latest portable Finance evidence against optional official data."""

import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not os.getenv("FINANCE_DATA_DIR"), reason="official local data not configured"
)
def test_official_retrained_bf16_evidence_and_training_identity():
    from evaluation.json_io import read_json
    from evaluation.query_encoder_report import verify_results
    from evaluation.query_encoder_training import training_data_audit

    data = Path(os.environ["FINANCE_DATA_DIR"])
    run = (
        Path(__file__).resolve().parents[2]
        / "docs/validation/2026-11-02/finance-retrain"
    )
    result = verify_results(data, run)
    assert result["queries"] == 721
    assert result["rows"] == 2884
    baseline = result["aggregates"]["baseline + Fixed alpha=0.84"]
    finance = result["aggregates"]["seed43 + Fixed alpha=0.84"]
    assert baseline["ndcg@10"] == pytest.approx(0.7909319389184100, abs=1e-12)
    assert finance["ndcg@10"] == pytest.approx(0.8145230714883048, abs=1e-12)
    assert result["aggregates"]["seed43 + Fixed"]["ndcg@10"] == pytest.approx(
        0.8149500101779286, abs=1e-12
    )
    frozen = read_json(run / "frozen.json")
    assert frozen["shared_alpha_dense"] == 0.84
    assert frozen["models"]["baseline"]["alpha_dense"] == 0.84
    assert frozen["models"]["seed43"]["alpha_dense"] == pytest.approx(0.83)
    protocol = read_json(run / "protocol.json")
    trained = protocol["encoders"]["seed43"]
    assert trained["selected_step"] == 4096
    assert trained["training"]["steps"] == 4096
    assert trained["training"]["seed"] == 43
    assert (
        trained["training"]["code"]["git_commit"]
        == "f49ec8b41a70ae663aec2837f51d69ec52e4c061"
    )
    assert trained["training"]["code"]["working_tree_dirty"] is False
    assert training_data_audit(data)["excluded_exact_duplicate_examples"] == 244

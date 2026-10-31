"""Phase 5 run identity 검사와 cross-model 집계를 모델 없이 검증합니다."""

import copy
import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("bm25s")
pytest.importorskip("faiss")

from data.bccard import prepare_bccard  # noqa: E402
from evaluation import cross_model as cm  # noqa: E402
from evaluation.enterprise import encode_bccard, run_enterprise_baselines  # noqa: E402
from evaluation.json_io import read_json, write_json  # noqa: E402
from evaluation.provenance import (  # noqa: E402
    ProtocolMismatch,
    compare_identities,
    run_identity,
)
from scripts import analyze_phase5  # noqa: E402
from tests.test_bccard import _write  # noqa: E402
from tests.test_enterprise_evaluation import _FakeEmbedder  # noqa: E402


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    data = tmp_path / "prepared"
    prepare_bccard(*_write(tmp_path / "raw"), data)
    monkeypatch.setattr("retrievers.dense.DenseEmbedder", _FakeEmbedder)
    reports = {}
    for encoder in ("bge", "kanana"):
        cache = tmp_path / f"cache-{encoder}"
        encode_bccard(data, cache, device="cpu", encoder=encoder)
        report = run_enterprise_baselines(data, cache, "test", warmup=1, repeats=1)
        path = tmp_path / f"{encoder}.json"
        write_json(path, report)
        reports[encoder] = path
    return data, reports, tmp_path


def test_identity_allows_model_change_and_refuses_other_qrels(prepared):
    data, reports, _ = prepared
    manifest = read_json(data / "manifest.json")
    bge, kanana = (read_json(reports[name]) for name in ("bge", "kanana"))
    a = run_identity(bge, "dense", manifest=manifest)
    b = run_identity(kanana, "dense", manifest=manifest)
    result = compare_identities(a, b)
    assert result["differences"]["model_name"] == [
        "BAAI/bge-m3",
        "kakaocorp/kanana-2-3b-embedding",
    ]
    assert "primary_slice_query_ids_sha256" not in result["unverified"]
    kanana["dataset"]["input_sha256"]["qrels_test.jsonl"] = "0" * 64
    with pytest.raises(ProtocolMismatch, match="qrels_sha256"):
        compare_identities(a, run_identity(kanana, "dense", manifest=manifest))


def test_identity_flags_reencoded_vectors_and_missing_primary(prepared):
    data, reports, _ = prepared
    manifest = read_json(data / "manifest.json")
    bge = read_json(reports["bge"])
    other = copy.deepcopy(bge)
    other["encoding"]["vector_sha256"]["document-vectors.npy"] = "f" * 64
    result = compare_identities(
        run_identity(bge, "dense", manifest=manifest),
        run_identity(other, "dense", manifest=manifest),
    )
    assert any("vector SHA-256" in warning for warning in result["warnings"])
    other["methods"]["dense"]["queries"] = other["methods"]["dense"]["queries"][:1]
    with pytest.raises(ProtocolMismatch, match="primary slice"):
        run_identity(other, "dense", manifest=manifest)


def test_identity_lists_fields_missing_from_old_reports(prepared):
    _, reports, _ = prepared
    bge = read_json(reports["bge"])
    result = compare_identities(
        run_identity(bge, "dense"), run_identity(bge, "score_fusion")
    )
    assert "primary_slice_query_ids_sha256" in result["unverified"]
    assert "config" in result["differences"]


def _row(ranked, ndcg, mode=None):
    row = {
        "results": [{"document_id": doc} for doc in ranked],
        "metrics": {"ndcg@10": ndcg},
    }
    if mode:
        row["mode"] = mode
    return row


def test_cross_model_counts():
    qrels = {"q1": {"d1": 3}, "q2": {"d2": 3}, "q3": {"d3": 3}}
    runs = {
        "bm25": {
            "q1": _row(["d1", "x"], 1.0),
            "q2": _row(["x", "y"], 0.0),
            "q3": _row(["d3"], 1.0),
        },
        "bge_dense": {
            "q1": _row(["x", "y"], 0.0),
            "q2": _row(["d2"], 1.0),
            "q3": _row(["d3"], 1.0),
        },
        "bge_fixed": {
            "q1": _row(["x", "d1"], 0.63),
            "q2": _row(["d2"], 1.0),
            "q3": _row(["x", "d3"], 0.63),
        },
        "kanana_dense": {
            "q1": _row(["y"], 0.0),
            "q2": _row(["x", "d2"], 0.63),
            "q3": _row(["d3"], 1.0),
        },
    }
    texts = {q: q for q in qrels}
    pairs = [("bm25", "bge_dense"), ("bm25", "kanana_dense")]
    rows = cm.query_table(runs, qrels, texts, list(qrels), pairs=pairs)
    assert rows[0]["bm25_rank"] == 1 and rows[0]["bge_dense_rank"] is None
    assert rows[0]["top10_overlap:bm25|bge_dense"] == 1
    assert rows[2]["top1_differs:bm25|bge_dense"] is False

    sparse = cm.sparse_complementarity(rows, "bm25", ["bge_dense", "kanana_dense"])
    assert sparse["bm25_hit@10_all_dense_miss"] == 1
    assert sparse["query_ids"]["bm25_hit@10_all_dense_miss"] == ["q1"]

    effect = cm.fusion_effect(rows, "bm25", "bge_dense", "bge_fixed")
    assert (effect["improved"], effect["regressed"], effect["tied"]) == (1, 1, 1)
    assert effect["rescued_from_dense_miss"] == 1
    assert effect["improved_bm25_rank"]["bm25_rank_1"] == 1
    assert effect["delta_histogram"]["+[0.5,1]"] == 1

    agreement = cm.encoder_agreement(rows, "bge_dense", "kanana_dense")
    assert agreement["hit@10"] == {
        "both": 2,
        "bge_dense_only": 0,
        "kanana_dense_only": 0,
        "neither": 1,
    }
    assert agreement["p@1"]["bge_dense_only"] == 1

    oracle = cm.method_oracle(rows, ("bm25", "bge_dense", "bge_fixed"), "bge_fixed")
    assert oracle["headroom_queries"] == ["q1", "q3"]
    assert oracle["choice_counts"] == {"bm25": 2, "bge_fixed": 1}


def test_adaptive_recovery_counts_mode_agreement():
    rows = [
        {
            "query_id": "q1",
            "fixed_ndcg@10": 0.5,
            "adaptive_ndcg@10": 0.5,
            "adaptive_mode": "balanced",
            "oracle_ndcg@10": 1.0,
            "oracle_mode": "keyword-heavy",
            "top1": True,
        },
        {
            "query_id": "q2",
            "fixed_ndcg@10": 1.0,
            "adaptive_ndcg@10": 1.0,
            "adaptive_mode": "balanced",
            "oracle_ndcg@10": 1.0,
            "oracle_mode": "balanced",
            "top1": False,
        },
    ]
    result = cm.adaptive_recovery(
        rows, fixed="fixed", adaptive="adaptive", oracle="oracle", disagreement="top1"
    )
    assert result["headroom_queries"] == 1
    assert result["router_improved_on_headroom"] == 0
    assert result["mode_agreement"] == 0.5
    assert result["by_bm25_dense_top1"]["top1_differs"]["oracle_minus_fixed"] == 0.5


def test_cli_compare_and_cross_model(prepared, capsys):
    data, reports, tmp_path = prepared
    runs = [f"--run=bge={reports['bge']}", f"--run=kanana={reports['kanana']}"]
    output = tmp_path / "paired.json"
    assert (
        analyze_phase5.main(
            [
                "compare",
                f"--data-dir={data}",
                *runs,
                "--pair=bge:dense,kanana:dense",
                "--resamples=20",
                f"--output={output}",
            ]
        )
        == 0
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    comparison = result["comparisons"][0]
    assert comparison["identity"]["differences"]["model_name"][1].startswith(
        "kakaocorp/"
    )
    assert set(comparison["metrics"]) == {"ndcg@10", "mrr@10", "recall@10", "p@1"}

    summary_path, rows_path = tmp_path / "cross.json", tmp_path / "rows.csv"
    columns = [
        "--column=bm25=bge:bm25",
        "--column=bge_dense=bge:dense",
        "--column=bge_fixed=bge:score_fusion",
        "--column=kanana_dense=kanana:dense",
        "--column=kanana_fixed=kanana:score_fusion",
    ]
    analyze_phase5.main(
        [
            "cross-model",
            f"--data-dir={data}",
            *runs,
            *columns,
            f"--output={summary_path}",
            f"--rows-output={rows_path}",
        ]
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["query_count"] == 2
    assert set(summary["fusion_effect"]) == {"bge", "kanana"}
    assert rows_path.read_text(encoding="utf-8").startswith("query_id,text,relevant")
    assert b"\r" not in rows_path.read_bytes()
    capsys.readouterr()


def test_cli_refuses_protocol_mismatch(prepared, capsys):
    data, reports, tmp_path = prepared
    broken = read_json(reports["kanana"])
    broken["conditions"]["top_k"] = 5
    path = tmp_path / "broken.json"
    write_json(path, broken)
    with pytest.raises(SystemExit):
        analyze_phase5.main(
            [
                "compare",
                f"--data-dir={data}",
                f"--run=bge={reports['bge']}",
                f"--run=other={path}",
                "--pair=bge:dense,other:dense",
                f"--output={tmp_path / 'out.json'}",
            ]
        )
    assert "top_k" in capsys.readouterr().err

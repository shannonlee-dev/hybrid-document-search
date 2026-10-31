"""CPU fixture evaluation produces a self-contained, independently replayable handoff."""

import shutil
from types import SimpleNamespace

import pytest

from evaluation.json_io import read_json, write_json
from evaluation.qdap_artifacts import evaluation_files, hashes, model_identity, sha

torch = pytest.importorskip("torch")


def test_evaluate_then_report_without_training_or_inference_artifacts(
    score_inputs, monkeypatch
):
    from evaluation import qdap_execution
    from evaluation.qdap_benchmark import prepare, rows
    from evaluation.qdap_report import enterprise_report
    from fusion.adaptive import FEATURES
    from fusion.qdap import QDAPSmall

    out = score_inputs / "run"
    prepare(score_inputs, out)
    model = score_inputs / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"fixture-only encoder")
    checkpoint = out / "small.pt"
    torch.save(QDAPSmall(2).state_dict(), checkpoint)
    router_path = score_inputs / "docs/experiments/adaptive/router.json"
    write_json(
        router_path,
        {
            "version": 1,
            "features": list(FEATURES),
            "weights": [0.2, 0.1, 0],
            "kind": "constant",
            "classes": [1],
            "mean": [0] * 10,
            "scale": [1] * 10,
        },
    )
    args = SimpleNamespace(
        inputs=score_inputs,
        output=out,
        model=model,
        device="cpu",
        model_name="BAAI/bge-m3",
        model_revision="1" * 40,
    )
    write_json(
        out / "selection.json",
        {
            "schema_version": 2,
            "run_id": "fixture-run",
            "selection_split": "validation",
            "protocol_sha256": sha(out / "protocol.json"),
            "paper_fixed_alpha": 0.84,
            "model_identity": model_identity(args),
            "evaluation_input_sha256": hashes(evaluation_files(args)),
            "selected": {
                variant: {
                    "epoch": 1,
                    "readout": "argmax",
                    "checkpoint": "small.pt",
                    "checkpoint_sha256": sha(checkpoint),
                }
                for variant in ("S", "L")
            },
        },
    )
    # Exercise actual scores, S checkpoint inference, ranking and output. Only the
    # expensive L encoder and the official fixture size constraint are substituted.
    data = score_inputs / "data/processed/bccard"
    docs, queries = rows(data / "corpus.jsonl"), rows(data / "queries_test.jsonl")
    monkeypatch.setattr(
        qdap_execution, "test_inputs", lambda args: (data, docs, queries)
    )
    monkeypatch.setattr(qdap_execution, "tokens", lambda *args: {"count": len(args[1])})
    monkeypatch.setattr(
        qdap_execution,
        "load_large",
        lambda *args: (lambda count: torch.full((count, 101), 1 / 101), None),
    )
    for name in (
        "train.npz",
        "validation.npz",
        "train-queries.json",
        "validation-queries.json",
        "validation-qrels.json",
    ):
        (out / name).unlink()
    for split in ("train", "validation"):
        for kind in ("queries", "qrels"):
            (data / f"{kind}_{split}.jsonl").unlink()
    qdap_execution.evaluate(args)
    with pytest.raises(ValueError, match="already evaluated"):
        qdap_execution.evaluate(args)
    source = out / "per-query.jsonl"
    original = source.read_bytes()
    assert len(rows(source)) == 16
    router_rows = [row for row in rows(source) if row["method"] == "Existing Router"]
    assert all(
        row["selected_mode"] == "balanced" and row["bm25_weight"] == 0.1
        for row in router_rows
    )
    checkpoint.unlink()
    shutil.rmtree(model)
    shutil.rmtree(score_inputs / "artifacts")
    shutil.rmtree(score_inputs / "docs")
    report = enterprise_report(score_inputs, source)
    assert len(report["methods"]) == 8
    assert report["conditions"]["official_baselines"] == []
    assert report["conditions"]["quality_run"] == "fixture-run"
    assert source.read_bytes() == original
    assert report["methods"]["qdap_s"]["slices"]["all"]["ndcg@10"] == pytest.approx(
        read_json(out / "test-summary.json")["rows"]["QDAP-S"]["ndcg@10"]
    )

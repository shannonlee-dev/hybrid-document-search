import copy
import hashlib
import json

import pytest

from evaluation.benchmark import benchmark_retriever
from evaluation.data import load_evaluation_dataset
from retrievers.base import SearchResult


@pytest.fixture
def curated_package(tmp_path):
    # A checkout must be verifiable without local raw reports or model files.
    names = (
        "dense/runtime.json",
        "dense/train/e5.json",
        "dense/train/bge.json",
        "dense/train/kure.json",
        "dense/train/comparison.csv",
        "dense/dev/e5.json",
        "dense/dev/bge.json",
        "dense/dev/kure.json",
        "dense/dev/comparison.csv",
        "retrieval/dev/results.json",
        "retrieval/dev/summary.csv",
    )
    files = {}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture\n")
        files[name] = hashlib.sha256(b"fixture\n").hexdigest()
    (tmp_path / "experiment.json").write_text(
        json.dumps({"status": "completed", "files_sha256": files})
    )
    return tmp_path


def test_curated_package_verifies_without_raw_reports(curated_package):
    from evaluation.experiment_results import verify_package

    verify_package(curated_package)


@pytest.mark.parametrize("mutation", ("extra_csv", "missing_summary", "changed"))
def test_curated_package_rejects_inventory_or_hash_changes(curated_package, mutation):
    from evaluation.experiment_results import verify_package

    manifest = curated_package / "experiment.json"
    report = json.loads(manifest.read_text())
    if mutation == "extra_csv":
        name = "retrieval/dev/queries.csv"
        (curated_package / name).write_bytes(b"{}\n")
        report["files_sha256"][name] = hashlib.sha256(b"{}\n").hexdigest()
    elif mutation == "missing_summary":
        name = "retrieval/dev/summary.csv"
        (curated_package / name).unlink()
        del report["files_sha256"][name]
    else:
        (curated_package / "retrieval/dev/summary.csv").write_bytes(b"changed\n")
    manifest.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="final package"):
        verify_package(curated_package)


@pytest.fixture
def result_fixture(evaluation_directory):
    class Retriever:
        def search(self, query, top_k):
            return [SearchResult("fixture-001#0", 1, 1)]

    dataset = load_evaluation_dataset(evaluation_directory)
    return dataset, benchmark_retriever(Retriever(), dataset)


def test_evidence_preserves_recomputable_results_without_document_excerpts(
    result_fixture, tmp_path
):
    from evaluation.experiment_results import _validate_method, _write_evidence
    from evaluation.latency import latency_summary
    from evaluation.metrics import evaluate_run

    dataset, result = result_fixture
    for query in result["queries"]:
        query["results"][0].update(title="fixture title", snippet="fixture excerpt")
    report = {"methods": {"dense": result}, "conditions": {"top_k": 10}}
    original = copy.deepcopy(report)
    path = tmp_path / "evidence.json"

    _write_evidence(path, report, dataset)

    saved = json.loads(path.read_text())
    assert report == original
    assert saved["conditions"] == {"top_k": 10}
    assert saved["qrels"] == dataset.qrels
    queries = saved["methods"]["dense"]["queries"]
    for query, raw in zip(
        queries, original["methods"]["dense"]["queries"], strict=True
    ):
        assert query["results"] == [
            {"document_id": "fixture-001#0", "rank": 1, "score": 1}
        ]
        assert query["text"] == raw["text"]
        assert query["latency_samples_ms"] == raw["latency_samples_ms"]
    _validate_method(saved["methods"]["dense"], dataset)
    # Readers can recompute metrics and latency from the exported JSON alone.
    run = {
        query["query_id"]: [SearchResult(**hit) for hit in query["results"]]
        for query in queries
    }
    assert evaluate_run(run, saved["qrels"]) == result["metrics"]
    assert (
        latency_summary(
            [sample for query in queries for sample in query["latency_samples_ms"]]
        )
        == result["latency"]
    )


def test_quality_and_latency_recomputed(result_fixture):
    from evaluation.experiment_results import _validate_method

    dataset, result = result_fixture
    _validate_method(result, dataset)
    for mutation in ("metrics", "latency", "samples", "id", "text"):
        broken = copy.deepcopy(result)
        if mutation == "metrics":
            broken["metrics"]["ndcg@10"] += 0.1
        elif mutation == "latency":
            broken["latency"]["mean_ms"] += 1
        elif mutation == "samples":
            broken["queries"][0]["latency_samples_ms"].pop()
        elif mutation == "id":
            broken["queries"][0]["results"][0]["document_id"] = "unknown"
        else:
            broken["queries"][0]["text"] = "changed"
        with pytest.raises(ValueError):
            _validate_method(broken, dataset)


def test_revision_mismatch_fails():
    from evaluation.dense_runtime import MODELS
    from evaluation.experiment_results import _validate_embedding

    model, revision = next(iter(MODELS.items()))
    config = {
        "model_name": model,
        "revision": revision,
        "batch_size": 1,
        "normalize_embeddings": True,
        "device": "cuda:0",
    }
    _validate_embedding(config, model)
    config["revision"] = "0" * 40
    with pytest.raises(ValueError, match="revision"):
        _validate_embedding(config, model)

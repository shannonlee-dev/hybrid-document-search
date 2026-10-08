"""Verify curated result integrity and recomputable metrics without raw artifacts."""

import copy
import hashlib
import json

import pytest

from evaluation.benchmark import benchmark_retriever
from evaluation.data import load_evaluation_dataset
from evaluation.experiment_results import PACKAGE_FILES
from retrievers.base import SearchResult


@pytest.fixture
def curated_package(tmp_path):
    """Provide a complete file inventory without raw reports or model files."""
    directory = tmp_path / "package"
    files = {}
    for name in PACKAGE_FILES.values():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture\n")
        files[name] = hashlib.sha256(b"fixture\n").hexdigest()
    (directory / "experiment.json").write_text(
        json.dumps({"status": "completed", "files_sha256": files})
    )
    return directory


def test_curated_package_verifies_without_raw_reports(curated_package):
    from evaluation.experiment_results import verify_package

    verify_package(curated_package)


@pytest.mark.parametrize(
    "mutation", ("extra_csv", "missing_summary", "deleted", "changed", "incomplete")
)
def test_curated_package_rejects_inventory_or_hash_changes(curated_package, mutation):
    from evaluation.experiment_results import verify_package

    manifest = curated_package / "experiment.json"
    report = json.loads(manifest.read_text())
    if mutation == "extra_csv":
        name = "retrieval/dev/queries.csv"
        (curated_package / name).write_bytes(b"{}\n")
        report["files_sha256"][name] = hashlib.sha256(b"{}\n").hexdigest()
    elif mutation in {"missing_summary", "deleted"}:
        name = "retrieval/dev/summary.csv"
        (curated_package / name).unlink()
        if mutation == "missing_summary":
            del report["files_sha256"][name]
    elif mutation == "incomplete":
        report["status"] = "running"
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
    # 내보낸 JSON만으로 품질 지표와 지연을 다시 계산할 수 있어야 한다.
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


def test_runtime_summary_preserves_measurements_without_bulk_samples_or_excerpts():
    from evaluation import experiment_results

    build = {
        "embedding_seconds": 12.5,
        "faiss_build_seconds": 0.1,
        "save_seconds": 0.2,
        "total_preparation_seconds": 13.0,
        "peak_gpu_allocated_bytes": 1024,
        "index_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
        "loaded_model_revision": "c" * 40,
        "build_runtime_threads": {"faiss_omp_max_threads": 2},
        "gpu_name": "fixture GPU",
        "cuda_version": "fixture CUDA",
        "cached_model_revision": "c" * 40,
    }
    restored = {
        "latency_mean_seconds": 0.002,
        "latency_p95_seconds": 0.003,
        "latency_samples": 3,
        "latency_samples_ms": [[1.0, 2.0, 3.0]],
        "latency_sample_axes": ["query_index", "repeat_index"],
        "validation_runtime_threads": {"faiss_omp_max_threads": 2},
        "smoke_validation": {
            "full_mapping_matches": True,
            "fp32_vectors": True,
            "l2_normalized": True,
            "document_embeddings_recreated": False,
            "top_k_preserved": True,
            "score_rtol": 1e-5,
            "score_atol": 1e-6,
            "query": "fixture query",
            "hits": [{"title": "fixture title", "snippet": "fixture text"}],
        },
    }
    download = {
        "model": "fixture-model",
        "revision": "c" * 40,
        "snapshot": "/local/cache/fixture-model",
        "download_seconds": 1.5,
        "global_cache_preserved": True,
    }
    original = copy.deepcopy((build, restored, download))

    summary = experiment_results._runtime_summary(
        "fixture-model", build, restored, download
    )

    assert (build, restored, download) == original
    assert summary["model_name"] == "fixture-model"
    assert summary["status"] == "completed"
    for key in (
        "embedding_seconds",
        "faiss_build_seconds",
        "save_seconds",
        "total_preparation_seconds",
        "peak_gpu_allocated_bytes",
        "index_sha256",
        "metadata_sha256",
        "loaded_model_revision",
        "build_runtime_threads",
    ):
        assert summary[key] == build[key]
    for key in (
        "latency_mean_seconds",
        "latency_p95_seconds",
        "latency_samples",
        "validation_runtime_threads",
    ):
        assert summary[key] == restored[key]
    assert summary["download"] == {
        "download_seconds": 1.5,
        "global_cache_preserved": True,
    }
    assert summary["smoke_validation"] == {
        "full_mapping_matches": True,
        "fp32_vectors": True,
        "l2_normalized": True,
        "document_embeddings_recreated": False,
        "top_k_preserved": True,
        "score_rtol": 1e-5,
        "score_atol": 1e-6,
    }
    for key in (
        "gpu_name",
        "cuda_version",
        "cached_model_revision",
        "latency_samples_ms",
        "latency_sample_axes",
    ):
        assert key not in summary

"""Verify final evaluation routing, provenance, metrics and dataset separation."""

import json
import sys
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from data.loader import Document
from evaluation import benchmark, final
from evaluation.data import file_sha256, load_evaluation_dataset
from retrievers.base import SearchResult
from scripts import evaluate, evaluate_final


class FixedRetriever:
    def search(self, query, top_k):
        return [SearchResult("fixture-001#0", 1, 1.0)]


@pytest.mark.parametrize("name", ["rrf", "score_fusion"])
def test_common_evaluation_keeps_explicit_fusion_names(
    evaluation_directory, monkeypatch, name
):
    built, fused = [], []

    def build(method, *_):
        built.append(method)
        retriever = FixedRetriever()
        retriever.embedder = SimpleNamespace(model=SimpleNamespace(device="cpu"))
        return retriever, {}

    class ScoreStub(FixedRetriever):
        def __init__(self, sparse, dense):
            fused.append((sparse, dense))

    monkeypatch.setattr(benchmark, "_build_retriever", build)
    monkeypatch.setitem(
        sys.modules,
        "fusion.score",
        SimpleNamespace(DEFAULT_BM25_WEIGHT=0.05, ScoreFusionRetriever=ScoreStub),
    )
    report = benchmark.run_benchmark(
        evaluation_directory, methods=("dense", name), dense_index="fixture", repeats=1
    )
    assert built == ["dense", "bm25"]
    assert list(report["methods"]) == ["dense", name]
    config = report["methods"][name]["config"]
    if name == "score_fusion":
        assert len(fused) == 1
        assert config["bm25_weight"] == 0.05
        assert config["candidate_scope"] == "full corpus"
        assert config["vector_correction"] is False
        assert "rank_constant" not in config
    else:
        assert config["rank_constant"] == 60


@pytest.mark.parametrize("method", ["rrf", "score_fusion"])
def test_new_methods_require_a_dense_index(method):
    with pytest.raises(ValueError, match="index"):
        benchmark.run_benchmark("missing", methods=(method,))
    assert evaluate._build_parser().parse_args(
        ["--methods", method, "--output-dir", "new"]
    ).methods == [method]


def test_final_benchmark_reuses_common_measurement_and_report(
    evaluation_directory, monkeypatch, tmp_path
):
    dataset = load_evaluation_dataset(evaluation_directory)
    monkeypatch.setattr(
        final,
        "load_final_dataset",
        lambda *a, **k: (dataset, {"name": "fixture"}, None, None),
    )
    monkeypatch.setattr(
        final,
        "_build_final_retrievers",
        lambda *a: (
            {name: FixedRetriever() for name in final.FINAL_METHODS},
            dict.fromkeys(final.FINAL_METHODS, 0),
            {name: {} for name in final.FINAL_METHODS},
        ),
    )
    report = final.run_final_benchmark("fixture", warmup=1, repeats=2)
    assert list(report["methods"]) == list(final.FINAL_METHODS)
    assert report["conditions"]["measurement_passes"] == 2
    assert "excludes model inference" in report["conditions"]["latency_scope"]
    for result in report["methods"].values():
        assert result["latency"]["sample_count"] == 4
        assert result["query_count"] == 2
        assert result["metrics"]["mrr@10"] == 0.5
    benchmark.write_report(report, tmp_path / "report")
    table = (tmp_path / "report/comparison.md").read_text(encoding="utf-8")
    assert "Dataset: synthetic-fixture" in table
    assert "excludes model inference" in table
    assert "Query-time latency만" not in table


@pytest.mark.parametrize(
    "options",
    [
        {"methods": ()},
        {"methods": ("dense", "dense")},
        {"methods": ("hybrid",)},
        {"warmup": True},
        {"repeats": 0},
    ],
)
def test_final_rejects_invalid_settings_before_loading(options):
    with pytest.raises(ValueError):
        final.run_final_benchmark("does-not-exist", **options)


def test_cache_hash_mismatch_fails_before_loading_vectors(tmp_path):
    directory = tmp_path / "fixture"
    directory.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"files": {"fixture/data.json": "wrong"}}', encoding="utf-8")
    (directory / "data.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        final.load_final_dataset(directory, manifest=manifest)


def test_cli_keeps_two_datasets_separate_and_defers_writes_until_success(
    monkeypatch, tmp_path
):
    pytest.importorskip("faiss")
    calls, writes = [], []

    def run(directory, **kwargs):
        calls.append((directory.name, kwargs))
        return {"environment": {}, "methods": {}}

    monkeypatch.setattr(evaluate_final, "run_final_benchmark", run)
    monkeypatch.setattr(evaluate_final, "write_report", lambda r, p: writes.append(p))
    assert evaluate_final.main(["--output-dir", str(tmp_path / "out")]) == 0
    assert [name for name, _ in calls] == ["miracl10k", "strategyqa"]
    assert writes == [tmp_path / "out/miracl10k", tmp_path / "out/strategyqa"]
    assert all(
        options["warmup"] == 1 and options["repeats"] == 5 for _, options in calls
    )
    writes.clear()

    def fail(directory, **kwargs):
        if directory.name == "strategyqa":
            raise ValueError("second dataset failed")
        return {"environment": {}, "methods": {}}

    monkeypatch.setattr(evaluate_final, "run_final_benchmark", fail)
    with pytest.raises(SystemExit):
        evaluate_final.main(["--output-dir", str(tmp_path / "new")])
    assert not writes


def test_cli_does_not_overwrite_or_repeat_datasets(tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    (directory / "keep.txt").write_text("keep", encoding="utf-8")
    for options in [[], ["--datasets", "miracl10k", "miracl10k"]]:
        with pytest.raises(SystemExit):
            evaluate_final.main(["--output-dir", str(directory), *options])
    assert (directory / "keep.txt").read_text(encoding="utf-8") == "keep"


@pytest.fixture
def frozen_fixture(tmp_path):
    pytest.importorskip("fusion.score")  # Requires the dense extra.
    np = pytest.importorskip("numpy")
    directory = tmp_path / "fixture"
    directory.mkdir()
    documents = [
        Document("d1", "서울은 대한민국 수도", "서울"),
        Document("d2", "한글 문자", None),
    ]
    data = {
        "name": "fixture",
        "documents": [asdict(doc) for doc in documents],
        "query_ids": ["train", "dev"],
        "queries": {"train": "한글", "dev": "서울"},
        "qrels": {"train": {"d2": 1}, "dev": {"d1": 1}},
        "splits": {"train": ["train"], "dev": ["dev"]},
    }
    np.save(directory / "document-vectors.npy", np.eye(2, dtype=np.float32))
    np.save(
        directory / "query-vectors.npy", np.array([[0, 1], [1, 0]], dtype=np.float32)
    )
    (directory / "data.json").write_text(json.dumps(data), encoding="utf-8")
    (directory / "encoding.json").write_text(
        json.dumps(
            {
                "vector_sha256": {
                    "document": file_sha256(directory / "document-vectors.npy"),
                    "query": file_sha256(directory / "query-vectors.npy"),
                }
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "release.json"

    def refresh():
        manifest.write_text(
            json.dumps(
                {
                    "files": {
                        f"fixture/{path.name}": file_sha256(path)
                        for path in directory.iterdir()
                    }
                }
            ),
            encoding="utf-8",
        )

    refresh()
    return directory, manifest, data, refresh


def test_real_retrievers_use_same_dev_ids_and_frozen_vectors(frozen_fixture):
    directory, manifest, _, _ = frozen_fixture
    report = final.run_final_benchmark(directory, manifest=manifest, repeats=1)
    assert report["dataset"]["query_ids"] == ["dev"]
    assert "encoding.json" in report["dataset"]["input_sha256"]
    for name, result in report["methods"].items():
        assert result["metrics"]["mrr@10"] == 1, name
        assert result["queries"][0]["results"][0]["document_id"] == "d1"
        assert result["latency"]["sample_count"] == 1


@pytest.mark.parametrize("mutation", ["overlap", "unknown_document", "bad_label"])
def test_final_cache_rejects_invalid_evaluation_references(frozen_fixture, mutation):
    directory, manifest, data, refresh = frozen_fixture
    if mutation == "overlap":
        data["splits"]["train"] = ["dev"]
    elif mutation == "unknown_document":
        data["qrels"]["dev"] = {"absent": 1}
    else:
        data["qrels"]["dev"] = {"d1": True}
    (directory / "data.json").write_text(json.dumps(data), encoding="utf-8")
    refresh()
    with pytest.raises(ValueError):
        final.load_final_dataset(directory, manifest=manifest)

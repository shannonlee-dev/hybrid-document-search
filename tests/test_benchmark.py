"""공통 검색기 재사용, timing 범위, 집계 및 파일 저장을 검증합니다."""

import csv
import json
from dataclasses import replace

import pytest

from evaluation import benchmark
from evaluation.data import load_evaluation_dataset
from retrievers.base import SearchResult


class FixtureRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        return [SearchResult("fixture-001#0", 1, 1)] if query == "수도" else []


def test_warmup_and_repeated_calls_are_timed_separately(
    evaluation_directory, monkeypatch
):
    dataset = load_evaluation_dataset(evaluation_directory)
    clock = iter([0, 0.050, 0.100, 0.101, 0.200, 0.203, 0.300, 0.305, 0.400, 0.407])
    monkeypatch.setattr(benchmark, "perf_counter", lambda: next(clock))
    retriever = FixtureRetriever()
    result = benchmark.benchmark_retriever(retriever, dataset, warmup=1, repeats=2)
    assert retriever.calls == [("수도", 10), ("한글", 10)] * 3
    assert result["warmup_seconds"] == pytest.approx(0.05)
    assert result["latency"] == pytest.approx(
        {"mean_ms": 4, "p95_ms": 6.7, "sample_count": 4}
    )
    assert result["queries"][0]["latency_samples_ms"] == pytest.approx([1, 5])
    assert result["queries"][1]["latency_samples_ms"] == pytest.approx([3, 7])
    assert result["metrics"] == {
        "recall@5": 0.5,
        "recall@10": 0.5,
        "mrr@10": 0.5,
        "ndcg@10": 0.5,
    }


@pytest.mark.parametrize(
    "samples, expected", [([1], 1), ([1, 2, 3, 4], 3.85), ([0, 10], 9.5)]
)
def test_p95_uses_linear_interpolation(samples, expected):
    assert benchmark.latency_summary(samples)["p95_ms"] == pytest.approx(expected)


@pytest.mark.parametrize("samples", [[], [True], [-1], [float("nan")], [float("inf")]])
def test_invalid_latency_samples(samples):
    with pytest.raises(ValueError, match="latency"):
        benchmark.latency_summary(samples)


@pytest.mark.parametrize(
    "options", [{"warmup": 0}, {"repeats": 0}, {"warmup": True}, {"repeats": 1.5}]
)
def test_invalid_repetition_options(evaluation_directory, options):
    with pytest.raises(ValueError):
        benchmark.benchmark_retriever(
            FixtureRetriever(), load_evaluation_dataset(evaluation_directory), **options
        )


@pytest.mark.parametrize(
    "results, message",
    [
        ([SearchResult("unknown#0", 1, 1)], "corpus"),
        ([SearchResult("fixture-001#0", 2, 1)], "rank"),
        ([SearchResult("fixture-001#0", 1, float("nan"))], "score"),
        ([SearchResult("fixture-001#0", 1, 1)] * 11, "Top-10"),
    ],
)
def test_invalid_retriever_results_fail(evaluation_directory, results, message):
    class InvalidRetriever:
        def search(self, query, top_k):
            return results

    with pytest.raises(ValueError, match=message):
        benchmark.benchmark_retriever(
            InvalidRetriever(), load_evaluation_dataset(evaluation_directory)
        )


def test_dataset_query_and_qrels_sets_must_match(evaluation_directory):
    dataset = load_evaluation_dataset(evaluation_directory)
    with pytest.raises(ValueError, match="집합"):
        benchmark.benchmark_retriever(FixtureRetriever(), replace(dataset, qrels={}))


def test_all_methods_share_dataset_and_hybrid_reuses_components(
    evaluation_directory, monkeypatch
):
    built = {}
    seen_datasets = []

    def build(name, dataset, index, device):
        built[name] = built.get(name, 0) + 1
        seen_datasets.append(dataset)
        retriever = FixtureRetriever()
        if name == "dense":
            from types import SimpleNamespace

            retriever.embedder = SimpleNamespace(model=SimpleNamespace(device="cpu"))
        return retriever, {"model": "fixture"}

    monkeypatch.setattr(benchmark, "_build_retriever", build)
    report = benchmark.run_benchmark(
        evaluation_directory,
        methods=("tfidf", "bm25", "dense", "hybrid"),
        dense_index="fixture-index",
        warmup=1,
        repeats=2,
    )
    assert built == {"tfidf": 1, "bm25": 1, "dense": 1}
    assert all(dataset is seen_datasets[0] for dataset in seen_datasets)
    assert tuple(report["methods"]) == benchmark.METHODS
    assert report["methods"]["hybrid"]["config"]["sparse"] == "bm25"
    assert report["methods"]["hybrid"]["config"]["rank_constant"] == 60
    assert report["conditions"]["top_k"] == 10
    assert report["dataset"]["query_count"] == 2
    for result in report["methods"].values():
        assert result["metrics"]["mrr@10"] == 0.5
        assert result["latency"]["sample_count"] == 4


def test_report_json_csv_and_markdown_are_consistent(
    evaluation_directory, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        benchmark, "_build_retriever", lambda *args: (FixtureRetriever(), {})
    )
    report = benchmark.run_benchmark(
        evaluation_directory, methods=("tfidf",), repeats=2
    )
    directory = tmp_path / "result"
    benchmark.write_report(report, directory)
    assert json.loads((directory / "results.json").read_text()) == report
    with (directory / "summary.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 1
    assert float(rows[0]["mrr@10"]) == 0.5
    assert rows[0]["sample_count"] == "4"
    with (directory / "queries.csv").open(encoding="utf-8", newline="") as source:
        assert len(list(csv.DictReader(source))) == 2
    with (directory / "latencies.csv").open(encoding="utf-8", newline="") as source:
        assert len(list(csv.DictReader(source))) == 4
    assert "Recall@100은 참고용" in (directory / "comparison.md").read_text()
    with pytest.raises(ValueError, match="output-dir"):
        benchmark.write_report(report, directory)


@pytest.mark.parametrize(
    "methods, index",
    [
        ((), None),
        (("tfidf", "tfidf"), None),
        (("other",), None),
        (("dense",), None),
        (("hybrid",), None),
    ],
)
def test_invalid_methods_fail_before_loading_data(methods, index):
    with pytest.raises(ValueError):
        benchmark.run_benchmark("does-not-exist", methods=methods, dense_index=index)


def test_dense_missing_index_files_fail_before_import(evaluation_directory, tmp_path):
    dataset = load_evaluation_dataset(evaluation_directory)
    with pytest.raises(ValueError, match="Dense 인덱스 파일"):
        benchmark._build_retriever("dense", dataset, tmp_path / "missing", "cpu")


def test_dense_index_document_mismatch_is_rejected(
    evaluation_directory, tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace

    dataset = load_evaluation_dataset(evaluation_directory)
    index = tmp_path / "index"
    index.mkdir()
    (index / "metadata.json").write_text("{}")
    (index / "index.faiss").write_bytes(b"fixture")

    class DenseStub:
        @classmethod
        def load(cls, directory, *, device):
            assert directory == index and device == "cpu"
            return SimpleNamespace(documents=())

    monkeypatch.setitem(
        sys.modules, "retrievers.dense", SimpleNamespace(DenseRetriever=DenseStub)
    )
    with pytest.raises(ValueError, match="공통 corpus"):
        benchmark._build_retriever("dense", dataset, index, "cpu")

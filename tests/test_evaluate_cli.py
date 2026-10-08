"""Verify Sparse evaluation commands, output files and input errors."""

import builtins
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts.evaluate import main


@pytest.fixture
def sparse_runtime(monkeypatch):
    """Record numerical thread limits while rejecting all Dense dependency imports."""
    from evaluation import runtime_config

    for name in runtime_config.THREAD_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(runtime_config, "_LIMITER", None)
    monkeypatch.setitem(sys.modules, "numpy", ModuleType("numpy"))
    scipy = ModuleType("scipy")
    scipy.linalg = ModuleType("scipy.linalg")
    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.linalg", scipy.linalg)
    pools = [{"num_threads": 1}]
    requested_limits = []

    def _record_limit(*, limits):
        pools[0]["num_threads"] = limits
        requested_limits.append(limits)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "threadpoolctl",
        SimpleNamespace(
            threadpool_info=lambda: pools,
            threadpool_limits=_record_limit,
        ),
    )
    original_import = builtins.__import__

    def _without_dense(name, *args, **kwargs):
        if name.split(".")[0] in {"sentence_transformers", "faiss", "torch"}:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _without_dense)
    return requested_limits, pools


@pytest.mark.parametrize("methods", [("tfidf",), ("bm25",), ("tfidf", "bm25")])
@pytest.mark.parametrize("device", [None, "cuda:0"])
def test_sparse_threads_without_dense_dependencies(
    evaluation_directory, tmp_path, monkeypatch, sparse_runtime, methods, device
):
    from evaluation import benchmark
    from retrievers.base import SearchResult
    from scripts import evaluate

    class FixtureRetriever:
        def search(self, query, top_k):
            return [SearchResult("fixture-001#0", 1, 1.0)]

    monkeypatch.setattr(
        benchmark, "_build_retriever", lambda *args: (FixtureRetriever(), {})
    )
    options = []
    original_benchmark = evaluate.run_benchmark

    def _run_benchmark(*args, **kwargs):
        options.append(kwargs)
        return original_benchmark(*args, **kwargs)

    monkeypatch.setattr(evaluate, "run_benchmark", _run_benchmark)
    directory = tmp_path / "sparse-threads"
    args = [
        "--data-dir",
        str(evaluation_directory),
        "--output-dir",
        str(directory),
        "--methods",
        *methods,
        "--threads",
        "2",
    ]
    if device is not None:
        args.extend(["--device", device])
    assert main(args) == 0
    report = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    assert set(report["methods"]) == set(methods)
    assert report["environment"]["effective_runtime"]["threadpools"] == [
        {"num_threads": 2}
    ]
    assert sparse_runtime[0] == [2]
    assert options[0]["strict_runtime"] is False


@pytest.mark.parametrize("method", ["dense", "hybrid"])
def test_dense_threads_keep_cuda_runtime(monkeypatch, tmp_path, method):
    from evaluation import runtime_config
    from scripts import evaluate

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", ModuleType("sentence_transformers")
    )
    calls = []
    runtime = {"device": "cuda:0"}
    monkeypatch.setattr(
        runtime_config,
        "configure_runtime",
        lambda threads, *, cuda: calls.append((threads, cuda)) or runtime,
    )
    options = []
    report = {"environment": {}, "methods": {}}
    monkeypatch.setattr(
        evaluate,
        "run_benchmark",
        lambda *args, **kwargs: options.append(kwargs) or report,
    )
    monkeypatch.setattr(evaluate, "write_report", lambda *args: None)
    assert (
        main(
            [
                "--methods",
                method,
                "--index",
                str(tmp_path / "index"),
                "--threads",
                "2",
                "--device",
                "cuda:0",
                "--output-dir",
                str(tmp_path / "result"),
            ]
        )
        == 0
    )
    assert calls == [(2, True)]
    assert options[0]["strict_runtime"] is True
    assert report["environment"]["effective_runtime"] == runtime


def test_sparse_runtime_rejects_ineffective_thread_limits(monkeypatch, sparse_runtime):
    from evaluation import runtime_config

    monkeypatch.setattr(
        sys.modules["threadpoolctl"], "threadpool_info", lambda: [{"num_threads": 4}]
    )
    with pytest.raises(RuntimeError, match="effective thread count"):
        runtime_config.configure_sparse_runtime(2)


def test_help_without_search_dependencies():
    result = subprocess.run(
        [sys.executable, "-B", "-S", "-m", "scripts.evaluate", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0
    assert "--methods" in result.stdout and "--index" in result.stdout
    assert "--output-dir" in result.stdout
    assert result.stderr == ""
    assert "\\n" not in result.stdout


def test_sparse_end_to_end_json_outputs(evaluation_directory, tmp_path, capsys):
    pytest.importorskip("sklearn")
    pytest.importorskip("bm25s")
    directory = tmp_path / "benchmark"
    assert (
        main(
            [
                "--data-dir",
                str(evaluation_directory),
                "--output-dir",
                str(directory),
                "--repeats",
                "2",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert set(summary) == {"tfidf", "bm25"}
    assert "검색기 준비" in captured.err and "평가:" in captured.err
    result = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    assert result["dataset"]["split"] == "dev"
    assert result["conditions"]["top_k"] == 10
    for method in summary:
        assert summary[method]["metrics"] == {
            "recall@5": 1,
            "recall@10": 1,
            "mrr@10": 1,
            "ndcg@10": 1,
        }
        assert summary[method]["latency"]["sample_count"] == 4
    assert set(path.name for path in directory.iterdir()) == {
        "results.json",
        "summary.csv",
        "queries.csv",
        "latencies.csv",
        "comparison.md",
    }


@pytest.mark.parametrize(
    "args, detail",
    [
        (["--repeats", "0"], "1 이상의 정수"),
        (["--warmup", "0"], "1 이상의 정수"),
        (["--split", "test"], "지원하지 않는"),
        (["--methods", "unknown"], "지원하지 않는"),
        (["--methods", "dense"], "--index"),
        (["--methods", "tfidf", "tfidf"], "중복"),
        (["--device", " "], "device"),
    ],
)
def test_input_errors_do_not_create_outputs(tmp_path, capsys, args, detail):
    directory = tmp_path / "result"
    with pytest.raises(SystemExit) as exc:
        main(["--output-dir", str(directory), *args])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert detail in captured.err
    assert "python -m scripts.evaluate --help" in captured.err
    assert not captured.out and not directory.exists()


def test_existing_output_is_not_overwritten(tmp_path, capsys):
    directory = tmp_path / "existing"
    directory.mkdir()
    original = directory / "results.json"
    original.write_text("keep")
    with pytest.raises(SystemExit) as exc:
        main(["--output-dir", str(directory)])
    assert exc.value.code == 2
    assert original.read_text() == "keep"
    assert "output-dir" in capsys.readouterr().err


def test_missing_data_reports_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--data-dir",
                str(tmp_path / "missing"),
                "--output-dir",
                str(tmp_path / "result"),
            ]
        )
    assert exc.value.code == 2
    assert "manifest" in capsys.readouterr().err

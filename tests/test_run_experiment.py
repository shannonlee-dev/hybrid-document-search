"""Experiment preflight accepts different environments while requiring CUDA."""

import runpy
import sys

import pytest

from scripts import run_experiment


def test_module_loads_without_fcntl(monkeypatch):
    monkeypatch.setitem(sys.modules, "fcntl", None)

    module = runpy.run_path(run_experiment.__file__)

    assert callable(module["main"])


@pytest.mark.parametrize(
    "python_version, kernel, gpu",
    [
        ((3, 13), "6.6.87-microsoft-standard-WSL2", "NVIDIA GeForce RTX 4060"),
        ((3, 12), "6.8.0-generic", "NVIDIA GeForce RTX 4060"),
        ((3, 12), "6.6.87-microsoft-standard-WSL2", "NVIDIA GeForce RTX 4090"),
    ],
)
def test_environment_accepts_cuda_without_hardware_or_os_restrictions(
    monkeypatch, python_version, kernel, gpu
):
    monkeypatch.setattr(run_experiment.sys, "version_info", python_version)
    monkeypatch.setattr(run_experiment.platform, "release", lambda: kernel)
    monkeypatch.setattr(run_experiment.platform, "platform", lambda: kernel)
    monkeypatch.setattr(
        run_experiment.platform,
        "python_version",
        lambda: f"{python_version[0]}.{python_version[1]}.0",
    )
    monkeypatch.setattr(
        run_experiment.importlib.metadata, "version", lambda name: "fixture-version"
    )

    def _configure_runtime(threads, *, cuda):
        if not cuda:
            raise RuntimeError("CUDA must be required for the experiment")
        return {"gpu": gpu, "cuda": "12.8", "device": "cuda:0"}

    monkeypatch.setattr(run_experiment, "configure_runtime", _configure_runtime)

    environment = run_experiment._environment()

    assert environment["python"] == f"{python_version[0]}.{python_version[1]}.0"
    assert environment["platform"] == kernel
    assert environment["gpu"] == gpu
    assert environment["cuda"] == "12.8"


def test_environment_does_not_fall_back_when_cuda_is_unavailable(monkeypatch):
    def _configure_runtime(threads, *, cuda):
        if cuda:
            raise RuntimeError("cuda:0 is required; CPU fallback is forbidden")
        return {"device": "cpu"}

    monkeypatch.setattr(run_experiment, "configure_runtime", _configure_runtime)

    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        run_experiment._environment()


@pytest.fixture
def local_pipeline(tmp_path, monkeypatch, evaluation_directory):
    """Exercise the real stage runner with local reports instead of GPU workers."""
    import json
    from types import SimpleNamespace

    from evaluation import benchmark
    from evaluation.json_io import write_json
    from retrievers.base import SearchResult

    root = tmp_path / "repository"
    root.mkdir()
    (root / "artifacts").mkdir()
    workspace = root / "artifacts/ko-miracl-full"
    for name in (
        "uv.lock",
        "scripts/prepare_dataset.py",
        "evaluation/constants.py",
        "evaluation/dense_runtime.py",
        "evaluation/runtime_config.py",
        "evaluation/latency.py",
        "scripts/benchmark_dense_runtime.py",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    monkeypatch.setattr(run_experiment, "ROOT", root)
    monkeypatch.setattr(run_experiment, "WORKSPACE", workspace)
    monkeypatch.setattr(run_experiment, "_code_hashes", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_experiment, "_environment", lambda: {"gpu": "fixture"})
    monkeypatch.setattr(run_experiment, "configure_runtime", lambda *a, **kw: {})
    monkeypatch.setattr(run_experiment, "validate_dataset", lambda *args: None)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda: SimpleNamespace(
                model_info=lambda model, revision: SimpleNamespace(sha=revision)
            )
        ),
    )

    # The experiment must also run from a checkout without Git publication checks.
    def _unexpected_git(*args, **kwargs):
        raise AssertionError("experiment attempted a Git command")

    monkeypatch.setattr(run_experiment.subprocess, "check_output", _unexpected_git)
    monkeypatch.chdir(tmp_path)
    calls = []
    failures = set()

    class Retriever:
        def search(self, query, top_k):
            return [SearchResult("fixture-001#0", 1, 1)]

    dataset = benchmark.load_evaluation_dataset(evaluation_directory)

    def _worker(directory, module, arguments):
        calls.append(directory.name)
        write_json(
            directory / "command.json",
            {"argv": list(map(str, arguments)), "exit_code": 0},
        )
        if directory.name in failures:
            raise RuntimeError("fixture worker failed")
        if module == "scripts.benchmark_dense_runtime":
            (directory / "stdout.log").write_text(json.dumps({"restored": True}))
        elif module == "scripts.evaluate":
            output = arguments[arguments.index("--output-dir") + 1]
            methods = arguments[
                arguments.index("--methods") + 1 : arguments.index("--index")
            ]
            benchmark.write_report(
                {
                    "dataset": {"split": dataset.split, **dataset.provenance},
                    "methods": {
                        method: {
                            "setup_seconds": 0.0,
                            **benchmark.benchmark_retriever(Retriever(), dataset),
                        }
                        for method in methods
                    },
                },
                output,
            )

    monkeypatch.setattr(run_experiment, "_run_command", _worker)
    return workspace, calls, failures


def test_local_completion_resume_and_fresh(local_pipeline, capsys):
    import json

    workspace, calls, _ = local_pipeline
    protected = workspace.parent / "other-index"
    protected.write_text("keep")
    assert run_experiment.main([]) == 0
    expected = ["data"]
    for alias in run_experiment.ALIASES:
        expected.extend([f"download-{alias}", f"build-{alias}", f"restore-{alias}"])
    expected.extend(
        f"{split}-{alias}"
        for split in ("train", "dev")
        for alias in run_experiment.ALIASES
    )
    expected.append("retrieval")
    assert calls == expected
    assert not (workspace / "package").exists()
    assert not (run_experiment.ROOT / "results").exists()
    checkpoint = json.loads((workspace / "checkpoint.json").read_text())
    assert list(checkpoint["steps"]) == expected
    assert all(row["status"] == "completed" for row in checkpoint["steps"].values())
    for stage in expected[-7:]:
        assert (workspace / stage / "evaluation/results.json").is_file()
        assert (workspace / stage / "evaluation/summary.csv").is_file()
    assert f"Experiment completed successfully: {workspace}" in capsys.readouterr().out
    calls.clear()
    assert run_experiment.main([]) == 0
    assert calls == []
    assert run_experiment.main(["--fresh"]) == 0
    assert calls == expected
    assert protected.read_text() == "keep"
    fresh = json.loads((workspace / "checkpoint.json").read_text())
    assert fresh["experiment_id"] != checkpoint["experiment_id"]
    assert not (run_experiment.ROOT / "results").exists()


def test_failed_stage_resumes_completed_work(local_pipeline):
    import json

    workspace, calls, failures = local_pipeline
    failures.add("train-bge")
    assert run_experiment.main([]) == 1
    checkpoint = json.loads((workspace / "checkpoint.json").read_text())
    assert checkpoint["steps"]["train-bge"]["status"] == "failed"
    assert (workspace / "failure.json").is_file()
    assert not (run_experiment.ROOT / "results").exists()
    completed = calls[:-1]
    calls.clear()
    failures.clear()
    assert run_experiment.main([]) == 0
    assert calls[0] == "train-bge"
    assert not set(calls) & set(completed)


def test_workspace_lock_prevents_concurrent_run(local_pipeline):
    workspace, calls, _ = local_pipeline
    if run_experiment.fcntl is None:
        pytest.skip("Unix workspace locking requires fcntl")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    with (workspace.parent / ".ko-miracl-full.lock").open("w") as handle:
        run_experiment.fcntl.flock(handle, run_experiment.fcntl.LOCK_EX)
        with pytest.raises(BlockingIOError):
            run_experiment.main([])
    assert calls == []

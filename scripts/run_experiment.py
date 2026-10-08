"""Fresh or checksum-validated resumable Ko-MIRACL GPU experiment."""

import argparse
import fcntl
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from data.preparation import PINNED_REVISION
from evaluation.data import file_sha256
from evaluation.experiment_results import CONDITIONS as CONDITIONS
from evaluation.experiment_results import (
    DATASET_SETTINGS,
    package_results,
    promote,
    read_json,
    validate_dataset,
    verify_package,
)
from evaluation.experiment_runner import Experiment, Step, hashes, reject_symlinks
from evaluation.json_io import write_json
from evaluation.runtime_config import THREAD_ENV, configure_runtime
from fusion.rrf import DEFAULT_RANK_CONSTANT
from retrievers.model_config import ALIASES, DEFAULT_MODEL_ALIAS, MODELS

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "artifacts/ko-miracl-full"
COMMAND = (
    "uv run --locked --extra sparse --extra dense python -m scripts.run_experiment"
)
_DOWNLOAD_ATTEMPTS = 3
_ERROR_TAIL_LENGTH = 6000


def _code_hashes(*directories, include_model_config=True):
    paths = [ROOT / "uv.lock", ROOT / "pyproject.toml"]
    if include_model_config and "retrievers" in directories:
        paths.append(ROOT / "config/dense_models.toml")
    for directory in directories:
        paths.extend(sorted((ROOT / directory).rglob("*.py")))
    paths.extend(
        [
            ROOT / "scripts/run_experiment.py",
            ROOT / "evaluation/experiment_runner.py",
            ROOT / "evaluation/json_io.py",
        ]
    )
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def _run_command(directory, module, arguments):
    argv = [sys.executable, "-m", module, *map(str, arguments)]
    environment = dict(os.environ, TOKENIZERS_PARALLELISM="false")
    environment.update({name: str(CONDITIONS["threads"]) for name in THREAD_ENV})
    with (
        (directory / "stdout.log").open("w") as stdout,
        (directory / "stderr.log").open("w") as stderr,
    ):
        attempts = (
            _DOWNLOAD_ATTEMPTS
            if module in {"scripts.prepare_dataset", "scripts.run_experiment"}
            else 1
        )
        for attempt in range(attempts):
            process = subprocess.run(
                argv, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr
            )
            if process.returncode == 0:
                break
            stderr.write(f"\nAttempt {attempt + 1}/{attempts} failed\n")
            stderr.flush()
    write_json(
        directory / "command.json", {"argv": argv, "exit_code": process.returncode}
    )
    if process.returncode:
        tail = (directory / "stderr.log").read_text()[-_ERROR_TAIL_LENGTH:]
        raise RuntimeError(f"{module} exited {process.returncode}: {tail}")


def _environment():
    runtime = configure_runtime(CONDITIONS["threads"], cuda=True)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": runtime["gpu"],
        "cuda": runtime["cuda"],
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "sentence-transformers",
                "faiss-cpu",
                "numpy",
                "scipy",
                "scikit-learn",
                "bm25s",
                "transformers",
                "huggingface-hub",
            )
        },
    }


def _ensure_old_results_preserved(fresh=False):
    """Refuse publication over results not already recoverable from local Git."""
    final = ROOT / "results"
    reject_symlinks(final)
    if not fresh and (final / "experiment.json").exists():
        try:
            verify_package(final)
            if hashes(final) == hashes(WORKSPACE / "package"):
                return
        except (ValueError, KeyError):
            pass
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", "results"],
        cwd=ROOT,
        text=True,
    )
    if status.strip():
        raise ValueError(
            "Preserve current results in a reviewed local commit before running"
        )
    tracked = subprocess.check_output(
        ["git", "ls-files", "--", "results"], cwd=ROOT, text=True
    ).splitlines()
    actual = {
        str(path.relative_to(ROOT)) for path in final.rglob("*") if path.is_file()
    }
    if actual - set(tracked):
        raise ValueError(
            "Untracked/ignored results must be preserved before replacement"
        )


def _make_steps(experiment, env):
    root = experiment.root
    data = root / "data/prepared"
    measurement_code = _code_hashes(
        "evaluation", "retrievers", "indexing", "fusion", "scripts", "data"
    )
    # Data/build fingerprints exclude evaluation/packaging code so changes to reports
    # do not require re-embedding successfully validated indexes.
    base = {"conditions": CONDITIONS, "environment": env}
    data_inputs = {
        "revision": PINNED_REVISION,
        "seed": DATASET_SETTINGS["seed"],
        "counts": [
            DATASET_SETTINGS["corpus_size"],
            DATASET_SETTINGS["train_queries"],
            DATASET_SETTINGS["dev_queries"],
        ],
        "code": {
            **_code_hashes("data"),
            "scripts/prepare_dataset.py": file_sha256(
                ROOT / "scripts/prepare_dataset.py"
            ),
        },
    }

    def _prepare(directory):
        _run_command(
            directory,
            "scripts.prepare_dataset",
            [
                "--download",
                "--revision",
                PINNED_REVISION,
                "--raw-dir",
                directory / "raw",
                "--output-dir",
                directory / "prepared",
            ],
        )
        validate_dataset(directory / "prepared", directory / "raw")

    steps = [Step("data", (), data_inputs, _prepare)]
    for alias, model in ALIASES.items():

        def _download(directory, model=model):
            _run_command(
                directory,
                "scripts.run_experiment",
                [
                    "--download-model",
                    model,
                    "--worker-output",
                    directory / "report.json",
                ],
            )

        steps.append(
            Step(
                f"download-{alias}",
                (),
                {
                    "model": model,
                    "revision": MODELS[model],
                    "lock": file_sha256(ROOT / "uv.lock"),
                },
                _download,
            )
        )
        index = root / f"build-{alias}/index"
        for phase, name, dependencies in (
            ("build", f"build-{alias}", ("data", f"download-{alias}")),
            ("validate", f"restore-{alias}", (f"build-{alias}",)),
        ):

            def _worker(directory, model=model, index=index, phase=phase):
                _run_command(
                    directory,
                    "scripts.benchmark_dense_runtime",
                    [
                        "--worker",
                        phase,
                        "--model",
                        model,
                        "--revision",
                        MODELS[model],
                        "--index-root",
                        index,
                        "--corpus",
                        data / "corpus.jsonl",
                        "--queries",
                        data / "queries_train.jsonl",
                        "--manifest",
                        data / "manifest.json",
                        "--threads",
                        CONDITIONS["threads"],
                    ],
                )
                write_json(
                    directory / "report.json", read_json(directory / "stdout.log")
                )

            # Model/revision are fingerprinted below; the default selection policy
            # only affects evaluation and must not invalidate explicit-model indexes.
            build_code = _code_hashes(
                "retrievers", "indexing", include_model_config=False
            )
            for name_code in (
                "evaluation/dense_runtime.py",
                "evaluation/runtime_config.py",
                "evaluation/latency.py",
                "scripts/benchmark_dense_runtime.py",
            ):
                build_code[name_code] = file_sha256(ROOT / name_code)
            steps.append(
                Step(
                    name,
                    dependencies,
                    {
                        **base,
                        "model": model,
                        "revision": MODELS[model],
                        "code": build_code,
                    },
                    _worker,
                )
            )
    for split in ("train", "dev"):
        for alias, model in ALIASES.items():

            def _evaluate(directory, split=split, alias=alias):
                _run_command(
                    directory,
                    "scripts.evaluate",
                    _evaluation_arguments(
                        root, data, directory, split, alias, ["dense"]
                    ),
                )

            steps.append(
                Step(
                    f"{split}-{alias}",
                    (f"restore-{alias}",),
                    {
                        **base,
                        "split": split,
                        "model": model,
                        "revision": MODELS[model],
                        "code": measurement_code,
                    },
                    _evaluate,
                )
            )

    def _retrieval(directory):
        _run_command(
            directory,
            "scripts.evaluate",
            _evaluation_arguments(
                root,
                data,
                directory,
                "dev",
                DEFAULT_MODEL_ALIAS,
                ["tfidf", "bm25", "dense", "hybrid"],
            ),
        )

    steps.append(
        Step(
            "retrieval",
            (f"restore-{DEFAULT_MODEL_ALIAS}",),
            {**base, "code": measurement_code, "rrf": DEFAULT_RANK_CONSTANT},
            _retrieval,
        )
    )
    dependencies = tuple(step.name for step in steps)

    def _package(directory):
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        provenance = {
            "conditions": CONDITIONS,
            "environment": env,
            "code_sha256": measurement_code,
            "code_commit": commit,
            "lock_sha256": file_sha256(ROOT / "uv.lock"),
            "command": COMMAND,
            "fresh_command": COMMAND + " --fresh",
            "index_regeneration": "--fresh rebuilds all three indexes from newly downloaded sources",
            "retrieval_command": read_json(root / "retrieval/command.json")["argv"],
            "stage_commands": {
                step.name: read_json(root / step.name / "command.json")
                for step in steps[:-1]
            },
            "preflight": read_json(root / "preflight.json"),
        }
        package_results(root, directory, experiment.identifier, provenance)

    steps.append(Step("package", dependencies, {"code": measurement_code}, _package))
    return steps


def _evaluation_arguments(root, data, directory, split, alias, methods):
    return [
        "--data-dir",
        data,
        "--split",
        split,
        "--methods",
        *methods,
        "--index",
        root / f"build-{alias}/index",
        "--device",
        CONDITIONS["device"],
        "--threads",
        CONDITIONS["threads"],
        "--warmup",
        CONDITIONS["warmup"],
        "--repeats",
        CONDITIONS["repeats"],
        "--output-dir",
        directory / "evaluation",
    ]


def _download_model(model, output):
    from huggingface_hub import HfApi, snapshot_download

    revision = MODELS[model]
    started = perf_counter()
    info = HfApi().model_info(model, revision=revision)
    if info.sha != revision:
        raise ValueError("remote snapshot revision mismatch")
    snapshot = snapshot_download(
        model,
        revision=revision,
        ignore_patterns=["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot"],
    )
    if Path(snapshot).name != revision:
        raise ValueError("downloaded snapshot revision mismatch")
    write_json(
        output,
        {
            "model": model,
            "revision": revision,
            "snapshot": snapshot,
            "download_seconds": perf_counter() - started,
            "global_cache_preserved": True,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--download-model", choices=tuple(MODELS), help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.download_model:
        _download_model(args.download_model, args.worker_output)
        return 0
    os.chdir(ROOT)
    for name in THREAD_ENV:
        os.environ[name] = str(CONDITIONS["threads"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    lock = WORKSPACE.parent / ".ko-miracl-full.lock"
    reject_symlinks(WORKSPACE.parent)
    lock.parent.mkdir(parents=True, exist_ok=True)
    reject_symlinks(lock)
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _ensure_old_results_preserved(args.fresh)
        changed_code = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "data",
                "retrievers",
                "indexing",
                "evaluation",
                "fusion",
                "scripts",
                "config",
                "pyproject.toml",
                "uv.lock",
            ],
            text=True,
        )
        if changed_code.strip():
            raise ValueError("Commit measurement code before starting the experiment")
        experiment = Experiment(WORKSPACE)
        try:
            env = _environment()
            if args.fresh:
                experiment.fresh()
            if not (experiment.root / "preflight.json").exists():
                from huggingface_hub import HfApi

                accessible = {
                    model: HfApi().model_info(model, revision=revision).sha
                    for model, revision in MODELS.items()
                }
                if accessible != MODELS:
                    raise ValueError("remote model revision preflight failed")
                write_json(
                    experiment.root / "preflight.json",
                    {
                        "environment": env,
                        "effective_runtime": configure_runtime(
                            CONDITIONS["threads"], cuda=True
                        ),
                        "accessible_model_revisions": accessible,
                        "command": COMMAND + (" --fresh" if args.fresh else ""),
                        "initial_head": subprocess.check_output(
                            ["git", "rev-parse", "HEAD"], text=True
                        ).strip(),
                    },
                )
            experiment.run(_make_steps(experiment, env))
            promote(experiment.root / "package", ROOT / "results")
            verify_package(ROOT / "results")
        except Exception as exc:
            write_json(
                experiment.root / "failure.json",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            print(str(exc), file=sys.stderr)
            return 1
    print("Verified complete experiment: results/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

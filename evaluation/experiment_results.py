"""Validate raw reports in artifacts and publish only curated result files."""

import copy
import csv
import ctypes
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from data.preparation import PINNED_REVISION
from evaluation.benchmark import _check_results
from evaluation.data import file_sha256, load_evaluation_dataset
from evaluation.experiment_runner import hashes, reject_symlinks
from evaluation.json_io import write_json
from evaluation.latency import DEFAULT_REPEATS, DEFAULT_WARMUP, latency_summary
from evaluation.metrics import (
    EVALUATION_TOP_K,
    METRIC_NAMES,
    evaluate_query,
    evaluate_run,
)
from evaluation.runtime_config import CUDA_DEVICE, DEFAULT_THREADS, THREAD_COUNT_KEYS
from fusion.rrf import DEFAULT_RANK_CONSTANT
from retrievers.base import SearchResult
from retrievers.model_config import ALIASES as ALIASES
from retrievers.model_config import (
    CONFIG_PATH,
    DEFAULT_MODEL,
    DEFAULT_MODEL_ALIAS,
    MODELS,
)

CONDITIONS = {
    "device": CUDA_DEVICE,
    "batch_size": 1,
    "dtype": "float32",
    "normalize_embeddings": True,
    "top_k": EVALUATION_TOP_K,
    "warmup": DEFAULT_WARMUP,
    "repeats": DEFAULT_REPEATS,
    "threads": DEFAULT_THREADS,
    "seed": 42,
}
DATASET_SETTINGS = {
    "corpus_size": 10000,
    "train_queries": 100,
    "dev_queries": 50,
    "seed": CONDITIONS["seed"],
}
RETRIEVAL_EXPORTS = ("summary.csv",)
_METRIC_TOLERANCE = 1e-10
# Linux renameat2 uses these values for the current directory and atomic exchange.
_AT_FDCWD = -100
_RENAME_EXCHANGE = 2


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_dataset(data, raw):
    datasets = {
        split: load_evaluation_dataset(data, split) for split in ("train", "dev")
    }
    manifest = read_json(data / "manifest.json")
    _require(
        manifest["dataset_revision"] == PINNED_REVISION, "dataset revision mismatch"
    )
    _require(
        manifest["settings"] == DATASET_SETTINGS,
        "dataset settings mismatch",
    )
    for name, digest in manifest["source_sha256"].items():
        _require(
            file_sha256(raw / PINNED_REVISION / name) == digest, "source hash mismatch"
        )
    for split, count in (
        ("train", DATASET_SETTINGS["train_queries"]),
        ("dev", DATASET_SETTINGS["dev_queries"]),
    ):
        _require(
            len(datasets[split].documents) == DATASET_SETTINGS["corpus_size"]
            and len(datasets[split].queries) == count,
            "dataset counts mismatch",
        )
    return datasets


def package_results(root, stage, identifier, provenance, *, retrieval_root=None):
    retrieval_root = (
        root / "retrieval" if retrieval_root is None else Path(retrieval_root)
    )
    data = root / "data/prepared"
    datasets = validate_dataset(data, root / "data/raw")
    raw_results = {}
    raw_runtime = {}
    runtime = {"models": []}
    for alias, model in ALIASES.items():
        build = read_json(root / f"build-{alias}/report.json")
        restored = read_json(root / f"restore-{alias}/report.json")
        _require(build["revision"] == MODELS[model], "build revision mismatch")
        _require(
            build["loaded_model_revision"] == MODELS[model],
            "build loaded revision mismatch",
        )
        _require(
            build["document_count"] == DATASET_SETTINGS["corpus_size"]
            and build["dtype"] == f"torch.{CONDITIONS['dtype']}",
            "build count/dtype mismatch",
        )
        _require(
            build["runtime_device"] == CONDITIONS["device"], "build device mismatch"
        )
        for counts in (
            build["build_runtime_threads"],
            restored["validation_runtime_threads"],
        ):
            _require(
                all(value == CONDITIONS["threads"] for value in counts.values()),
                "build/restore threads mismatch",
            )
        smoke = restored["smoke_validation"]
        _require(
            smoke["full_mapping_matches"]
            and smoke["top_k_preserved"]
            and not smoke["document_embeddings_recreated"],
            "restore verification failed",
        )
        _require(
            smoke["revision"] == smoke["loaded_model_revision"] == MODELS[model],
            "restore revision mismatch",
        )
        runtime["models"].append(
            _runtime_summary(
                model,
                build,
                restored,
                read_json(root / f"download-{alias}/report.json"),
            )
        )
        raw_runtime[alias] = {
            phase: {
                "path": str(root / f"{phase}-{alias}/report.json"),
                "sha256": file_sha256(root / f"{phase}-{alias}/report.json"),
            }
            for phase in ("build", "restore", "download")
        }
    write_json(stage / "dense/runtime.json", runtime)
    for split in ("train", "dev"):
        rows = {}
        for alias, model in ALIASES.items():
            source = root / f"{split}-{alias}/evaluation/results.json"
            report = read_json(source)
            _validate_report(report, datasets[split], ("dense",), model)
            raw_results[f"dense/{split}/{alias}.json"] = {
                "path": str(source),
                "sha256": file_sha256(source),
            }
            _write_evidence(
                stage / f"dense/{split}/{alias}.json", report, datasets[split]
            )
            rows[alias] = report["methods"]["dense"]
        _write_comparison(stage / f"dense/{split}/comparison.csv", rows)
    source = retrieval_root / "evaluation/results.json"
    report = read_json(source)
    _validate_report(
        report, datasets["dev"], ("tfidf", "bm25", "dense", "hybrid"), DEFAULT_MODEL
    )
    destination = stage / "retrieval/dev"
    destination.mkdir(parents=True, exist_ok=True)
    for name in RETRIEVAL_EXPORTS:
        shutil.copyfile(retrieval_root / "evaluation" / name, destination / name)
    raw_results["retrieval/dev/results.json"] = {
        "path": str(source),
        "sha256": file_sha256(source),
    }
    _write_evidence(destination / "results.json", report, datasets["dev"])
    write_json(
        stage / "experiment.json",
        {
            "experiment_id": identifier,
            "status": "completed",
            **provenance,
            "dataset_revision": PINNED_REVISION,
            "manifest_sha256": file_sha256(data / "manifest.json"),
            "models": MODELS,
            "default_model": DEFAULT_MODEL,
            "selection_policy": "config_default",
            "model_config": {
                "path": "config/dense_models.toml",
                "sha256": file_sha256(CONFIG_PATH),
                "default_alias": DEFAULT_MODEL_ALIAS,
            },
            "files_sha256": hashes(stage),
            "raw_results": raw_results,
            "raw_runtime": raw_runtime,
            "validations": [
                "source/output hashes",
                "10000/100/50 dataset and qrels",
                "three pinned FP32 CUDA builds and subprocess restores",
                "no document re-embedding on restore",
                "six dense quality runs",
                "four retrieval methods",
                "recomputed metrics and latency samples",
                "effective library threads",
                "raw JSON preserved; review exports omit hit title/snippet and include qrels",
            ],
            "scope": "Reused Ko-MIRACL subset/dev reproducibility experiment; not independent testing or full MIRACL.",
        },
    )
    verify_package(stage)


def verify_package(directory):
    report = read_json(directory / "experiment.json")
    actual = hashes(directory)
    actual.pop("experiment.json")
    _require(
        report["status"] == "completed" and actual == report["files_sha256"],
        "final package hash mismatch",
    )
    required = {"dense/runtime.json"}
    for split in ("train", "dev"):
        required.update(
            f"dense/{split}/{name}"
            for name in (*[f"{alias}.json" for alias in ALIASES], "comparison.csv")
        )
    required.update(
        f"retrieval/dev/{name}" for name in (*RETRIEVAL_EXPORTS, "results.json")
    )
    _require(set(actual) == required, "final package file inventory mismatch")


def promote(stage, final):
    """Linux renameat2 exchange makes replacement visible as one complete set."""
    verify_package(stage)
    reject_symlinks(final)
    if final.exists() and hashes(final) == hashes(stage):
        return
    temporary = Path(tempfile.mkdtemp(prefix=".ko-miracl-publish-", dir=final.parent))
    try:
        shutil.copytree(stage, temporary, dirs_exist_ok=True)
        verify_package(temporary)
        if final.exists():
            rename = ctypes.CDLL(None, use_errno=True).renameat2
            rename.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            if (
                rename(
                    _AT_FDCWD,
                    os.fsencode(temporary),
                    _AT_FDCWD,
                    os.fsencode(final),
                    _RENAME_EXCHANGE,
                )
                != 0
            ):
                raise OSError(ctypes.get_errno(), "atomic results exchange failed")
        else:
            temporary.rename(final)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _validate_report(report, dataset, methods, model):
    _require(set(report["methods"]) == set(methods), "missing/unexpected methods")
    _require(
        report["dataset"] == {"split": dataset.split, **dataset.provenance},
        "dataset provenance mismatch",
    )
    for key, expected in {
        "top_k": CONDITIONS["top_k"],
        "warmup_passes": CONDITIONS["warmup"],
        "measurement_passes": CONDITIONS["repeats"],
        "query_order": [q.query_id for q in dataset.queries],
    }.items():
        _require(report["conditions"][key] == expected, f"condition mismatch: {key}")
    runtime = report["environment"]["effective_runtime"]
    _require(runtime["device"] == CONDITIONS["device"], "CUDA runtime mismatch")
    for key in THREAD_COUNT_KEYS:
        _require(runtime[key] == CONDITIONS["threads"], f"threads mismatch: {key}")
    _require(
        all(
            pool["num_threads"] == CONDITIONS["threads"]
            for pool in runtime["threadpools"]
        ),
        "BLAS threads mismatch",
    )
    for method, result in report["methods"].items():
        _validate_method(result, dataset)
        if method in {"dense", "hybrid"}:
            config = (
                result["config"] if method == "dense" else result["config"]["dense"]
            )
            _validate_embedding(config["embedding"], model)
            _require(
                config["runtime_device"] == CONDITIONS["device"],
                "loaded device mismatch",
            )
            _require(
                config["runtime_dtype"] == f"torch.{CONDITIONS['dtype']}",
                "FP32 mismatch",
            )
            _require(
                config["loaded_model_revision"] == MODELS[model],
                "loaded revision mismatch",
            )
            index = Path(config["index"])
            _require(
                config["index_sha256"] == file_sha256(index / "index.faiss"),
                "index SHA mismatch",
            )
            _require(
                config["metadata_sha256"] == file_sha256(index / "metadata.json"),
                "metadata SHA mismatch",
            )
        if method == "hybrid":
            _require(
                result["config"]["rank_constant"] == DEFAULT_RANK_CONSTANT,
                "RRF mismatch",
            )


def _runtime_summary(model, build, restored, download):
    """Keep build/restore measurements while raw samples stay in artifacts."""
    return {
        "model_name": model,
        "status": "completed",
        **{
            key: value
            for key, value in build.items()
            if key not in {"gpu_name", "cuda_version", "cached_model_revision"}
        },
        **{
            key: value
            for key, value in restored.items()
            if key
            not in {"latency_samples_ms", "latency_sample_axes", "smoke_validation"}
        },
        "smoke_validation": {
            key: value
            for key, value in restored["smoke_validation"].items()
            if key not in {"query", "hits"}
        },
        "download": {
            key: download[key] for key in ("download_seconds", "global_cache_preserved")
        },
    }


def _write_evidence(path, report, dataset):
    """Keep remote metric/ranking evidence without repeating corpus excerpts."""
    evidence = copy.deepcopy(report)
    evidence["qrels"] = dataset.qrels
    evidence["export"] = {
        "kind": "review_evidence",
        "omitted_result_fields": ["title", "snippet"],
    }
    for result in evidence["methods"].values():
        for query in result["queries"]:
            for hit in query["results"]:
                hit.pop("title", None)
                hit.pop("snippet", None)
    write_json(path, evidence)


def _validate_embedding(config, model):
    for key, expected in {
        "model_name": model,
        "revision": MODELS[model],
        "batch_size": CONDITIONS["batch_size"],
        "normalize_embeddings": CONDITIONS["normalize_embeddings"],
        "device": CONDITIONS["device"],
    }.items():
        _require(config.get(key) == expected, f"embedding {key} mismatch")


def _validate_method(result, dataset):
    _require(result["query_count"] == len(dataset.queries), "query count mismatch")
    _require(len(result["queries"]) == len(dataset.queries), "raw query count mismatch")
    corpus = {doc.document_id for doc in dataset.documents}
    run, samples = {}, []
    for actual, query in zip(result["queries"], dataset.queries, strict=True):
        _require(
            actual["query_id"] == query.query_id and actual["text"] == query.text,
            "query ID/order/text mismatch",
        )
        hits = [SearchResult(**hit) for hit in actual["results"]]
        _check_results(hits, dataset.qrels[query.query_id], corpus)
        _require(
            len(actual["latency_samples_ms"]) == CONDITIONS["repeats"],
            "latency sample count mismatch",
        )
        _close_mapping(
            actual["metrics"],
            evaluate_query(hits, dataset.qrels[query.query_id]),
            "query metrics",
        )
        _close_mapping(
            actual["latency"],
            latency_summary(actual["latency_samples_ms"]),
            "query latency",
        )
        samples.extend(actual["latency_samples_ms"])
        run[query.query_id] = hits
    _close_mapping(
        result["metrics"], evaluate_run(run, dataset.qrels), "aggregate metrics"
    )
    _close_mapping(result["latency"], latency_summary(samples), "aggregate latency")


def _write_comparison(path, rows):
    fields = ("model", *METRIC_NAMES, "mean_ms", "p95_ms", "sample_count")
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for alias, result in rows.items():
            writer.writerow(
                {"model": ALIASES[alias], **result["metrics"], **result["latency"]}
            )


def _close_mapping(actual, expected, name):
    _require(set(actual) == set(expected), f"{name}: keys differ")
    _require(
        all(
            math.isclose(
                actual[k], v, rel_tol=_METRIC_TOLERANCE, abs_tol=_METRIC_TOLERANCE
            )
            for k, v in expected.items()
        ),
        f"{name}: recomputation differs",
    )


def _require(condition, message):
    if not condition:
        raise ValueError(message)

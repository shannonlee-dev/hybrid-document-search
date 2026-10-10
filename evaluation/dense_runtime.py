"""Measure Dense build and search runtimes and verify input and index integrity."""

import hashlib
import json
import math
import os
import platform
import re
import subprocess
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from data.loader import load_prepared_documents
from evaluation.constants import DEFAULT_THREADS, MILLISECONDS_PER_SECOND
from evaluation.json_io import write_json
from evaluation.latency import measure_query_latency
from retrievers.model_config import COMMIT_SHA_PATTERN
from retrievers.model_config import MODELS as MODELS

MEASUREMENTS = (
    "embedding_seconds",
    "faiss_build_seconds",
    "total_preparation_seconds",
    "latency_mean_seconds",
    "latency_p95_seconds",
    "latency_samples",
)
_RESTORE_RTOL = 1e-5
_RESTORE_ATOL = 1e-6
_BUILD_REPORT_FILENAME = "benchmark_build.json"


@dataclass
class DenseRuntimeConfig:
    """Dense runtime and measurement settings shared by the CLI and workers."""

    corpus: Path
    queries: Path
    manifest: Path
    device: str
    batch_size: int
    top_k: int
    warmup: int
    repeats: int
    index_root: Path
    threads: int = DEFAULT_THREADS
    model: tuple[str, ...] | None = None
    revision: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.revision is not None:
            if self.model is None or len(self.revision) != len(self.model):
                raise ValueError("--revision requires one SHA per explicit --model")
            if any(
                not isinstance(sha, str) or not re.fullmatch(COMMIT_SHA_PATTERN, sha)
                for sha in self.revision
            ):
                raise ValueError("revision must be a fixed 40-character commit SHA")

    def revision_for(self, model_name: str) -> str | None:
        if self.revision is not None:
            return self.revision[self.model.index(model_name)]
        return MODELS.get(model_name)


def build(config: DenseRuntimeConfig, model_name: str, index: Path):
    """Build and save an index, recording timings and reference results for restore checks."""
    started = perf_counter()
    from retrievers.dense import DenseConfig, DenseRetriever

    documents, queries, provenance = check_inputs(config)
    revision = config.revision_for(model_name)
    if revision is None and not Path(model_name).is_dir():
        raise ValueError("remote model requires a fixed revision")
    with _checkpoint_revisions() as checkpoint_revisions:
        retriever = DenseRetriever.build(
            documents,
            DenseConfig(
                model_name=model_name,
                revision=revision,
                device=config.device,
                batch_size=config.batch_size,
            ),
        )
    loaded_revision = _verify_loaded_revision(retriever, revision, checkpoint_revisions)
    save_started = perf_counter()
    retriever.save(index)
    save_seconds = perf_counter() - save_started
    _verify_saved_revision(index, model_name, revision)
    elapsed = perf_counter() - started
    result = {
        **retriever.build_timings,
        "save_seconds": save_seconds,
        "total_preparation_seconds": elapsed,
        "document_count": retriever.index.count,
        "revision": revision,
        "loaded_model_revision": loaded_revision,
    }
    result["build_runtime_threads"] = _runtime_threads()
    result["runtime_device"] = (
        str(retriever.embedder.model.device)
        if hasattr(retriever.embedder.model, "device")
        else config.device
    )
    result["index_sha256"] = _sha256(index / "index.faiss")
    result["metadata_sha256"] = _sha256(index / "metadata.json")
    # 복원 비교 기준은 메모리 인덱스에서 얻고 준비 시간에서 제외한다.
    baseline = {
        "provenance": provenance,
        "model_name": model_name,
        "revision": revision,
        "top_k": config.top_k,
        "query": queries[0],
        "hits": [asdict(hit) for hit in retriever.search(queries[0], config.top_k)],
        "timings": result,
    }
    model = retriever.embedder.model
    if hasattr(model, "max_seq_length"):
        result["max_seq_length"] = model.max_seq_length
    if hasattr(model, "parameters"):
        parameter = next(model.parameters(), None)
        result["dtype"] = str(parameter.dtype) if parameter is not None else None
    if not Path(model_name).is_dir():
        from huggingface_hub import try_to_load_from_cache

        cached_config = try_to_load_from_cache(
            model_name, "config.json", revision=retriever.config.revision
        )
        if isinstance(cached_config, str):
            result["cached_model_revision"] = Path(cached_config).parent.name
    if config.device.startswith("cuda"):
        import torch

        result["gpu_name"] = torch.cuda.get_device_name(config.device)
        result["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(
            config.device
        )
        result["cuda_version"] = torch.version.cuda
    write_json(index / _BUILD_REPORT_FILENAME, baseline)
    return result


def validate(config: DenseRuntimeConfig, index: Path):
    """Verify index restoration without re-embedding documents and measure search latency."""
    from retrievers.dense import DenseEmbedder, DenseRetriever

    documents, queries, provenance = check_inputs(config)
    baseline = json.loads((index / _BUILD_REPORT_FILENAME).read_text(encoding="utf-8"))
    if baseline["provenance"] != provenance or baseline["top_k"] != config.top_k:
        raise ValueError("build/validation inputs changed")
    model_name = config.model[0] if config.model else baseline["model_name"]
    if model_name != baseline["model_name"]:
        raise ValueError("requested model differs from build")
    revision = config.revision_for(model_name)
    if "revision" not in baseline or baseline["revision"] != revision:
        raise ValueError("requested revision differs from build revision")
    _verify_saved_revision(index, model_name, revision)
    with patch.object(
        DenseEmbedder,
        "encode_documents",
        side_effect=RuntimeError("document re-embedding forbidden"),
    ):
        restored = DenseRetriever.load(index, device=config.device)
        if restored.index.count != len(documents):
            raise ValueError("loaded index document count mismatch")
        if restored.documents != tuple(documents):
            raise ValueError("loaded document mapping differs from corpus")
        if restored.config.model_name != baseline["model_name"]:
            raise ValueError("loaded model differs from build")
        if restored.config.revision != revision:
            raise ValueError("loaded configuration revision differs from build")
        import numpy as np

        vectors = restored.index.index.reconstruct_n(0, restored.index.count)
        if vectors.dtype != np.float32:
            raise ValueError("restored index is not FP32")
        if restored.config.normalize_embeddings and not np.allclose(
            np.linalg.norm(vectors, axis=1), 1.0, rtol=_RESTORE_RTOL, atol=_RESTORE_ATOL
        ):
            raise ValueError("restored index vectors are not L2 normalized")
        with _checkpoint_revisions() as checkpoint_revisions:
            hits = [
                asdict(hit) for hit in restored.search(baseline["query"], config.top_k)
            ]
        loaded_revision = _verify_loaded_revision(
            restored, revision, checkpoint_revisions
        )
        if len(hits) != min(config.top_k, len(documents)):
            raise ValueError("unexpected search result count")
        for before, after in zip(baseline["hits"], hits, strict=True):
            if any(
                before[key] != after[key]
                for key in ("document_id", "rank", "title", "snippet")
            ):
                raise ValueError("Top-K mapping/rank changed after load")
            if not math.isclose(
                before["score"],
                after["score"],
                rel_tol=_RESTORE_RTOL,
                abs_tol=_RESTORE_ATOL,
            ):
                raise ValueError("Top-K score changed after load")
        measurement = measure_query_latency(
            restored,
            queries,
            top_k=config.top_k,
            warmup=config.warmup,
            repeats=config.repeats,
        )
    latency = measurement.latency
    return {
        # 기존 Dense runtime 보고서는 지연을 초 단위로 저장한다.
        "latency_mean_seconds": latency["mean_ms"] / MILLISECONDS_PER_SECOND,
        "latency_p95_seconds": latency["p95_ms"] / MILLISECONDS_PER_SECOND,
        "latency_samples": latency["sample_count"],
        # 중복 질의도 0부터 시작하는 입력 위치별로 구분한다.
        "latency_samples_ms": measurement.samples_ms,
        "latency_sample_axes": ["query_index", "repeat_index"],
        "validation_runtime_threads": _runtime_threads(),
        "smoke_validation": {
            "document_count": restored.index.count,
            "full_mapping_matches": True,
            "fp32_vectors": True,
            "l2_normalized": restored.config.normalize_embeddings,
            "document_embeddings_recreated": False,
            "top_k_preserved": True,
            "revision": revision,
            "loaded_model_revision": loaded_revision,
            "score_rtol": _RESTORE_RTOL,
            "score_atol": _RESTORE_ATOL,
            "query": baseline["query"],
            "hits": hits,
        },
    }


def create_report(
    config: DenseRuntimeConfig, provenance: dict[str, object], document_count: int
):
    """Create a runtime report with environment details and initial model states."""
    report = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "scope": "runtime/index only; no quality evaluation",
        "provenance": provenance,
        "code": _code_provenance(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in ("torch", "sentence-transformers", "faiss-cpu", "numpy")
            },
            "threads": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        },
        "warmup": config.warmup,
        "repeats": config.repeats,
        "models": [],
    }
    models = config.model or MODELS
    for model in models:
        index = config.index_root / model.replace("/", "--")
        report["models"].append(
            {
                "model_name": model,
                "revision": config.revision_for(model),
                "device": config.device,
                "batch_size": config.batch_size,
                "document_count": document_count,
                "top_k": config.top_k,
                "index": str(index),
                "status": "not_run",
                **dict.fromkeys(MEASUREMENTS),
            }
        )
    return report


def check_inputs(config: DenseRuntimeConfig):
    """Verify input hashes; return documents, queries and provenance."""
    if config.corpus.name == config.queries.name:
        raise ValueError(
            f"input basename collision: {config.corpus.name!r}; "
            f"corpus={config.corpus}, queries={config.queries}; "
            "manifest output_sha256 cannot distinguish these inputs"
        )
    manifest = json.loads(config.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{config.manifest}: manifest must be a JSON object")
    hashes = {path.name: _sha256(path) for path in (config.corpus, config.queries)}
    for name, digest in hashes.items():
        if manifest.get("output_sha256", {}).get(name) != digest:
            raise ValueError(f"manifest hash mismatch: {name}")
    documents = load_prepared_documents(config.corpus)
    records = []
    for line_number, line in enumerate(
        config.queries.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        location = f"{config.queries}:{line_number}:"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{location} invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{location} query record must be a JSON object")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"{location} query text must be a non-empty string")
        records.append(row)
    if not records:
        raise ValueError(f"{config.queries}: queries must contain non-empty text")
    return (
        documents,
        [row["text"] for row in records],
        {
            "dataset_revision": manifest["dataset_revision"],
            "manifest_sha256": _sha256(config.manifest),
            "input_sha256": hashes,
            "corpus": str(config.corpus),
            "queries": str(config.queries),
            "query_count": len(records),
        },
    )


@contextmanager
def _checkpoint_revisions():
    """Capture revisions from snapshot paths used to load Transformer weights.

    Checkpoint paths provide evidence when config._commit_hash is absent.
    """
    from transformers import modeling_utils

    revisions = []
    resolver = getattr(modeling_utils, "_get_resolved_checkpoint_files", None)
    if resolver is None:
        yield revisions
        return

    def _capture(*args, **kwargs):
        result = resolver(*args, **kwargs)
        for filename in result[0] or []:
            parts = Path(filename).parts
            revision = next(
                (
                    parts[i + 1]
                    for i, part in enumerate(parts[:-1])
                    if part == "snapshots"
                ),
                None,
            )
            revisions.append(revision)
        return result

    with patch.object(modeling_utils, "_get_resolved_checkpoint_files", _capture):
        yield revisions


def _verify_loaded_revision(retriever, revision, checkpoint_revisions):
    if Path(retriever.config.model_name).is_dir() and revision is None:
        return None
    model = retriever.embedder.model
    module = model._first_module() if hasattr(model, "_first_module") else None
    transformer = getattr(module, "auto_model", None)
    loaded_revision = getattr(
        getattr(transformer, "config", None), "_commit_hash", None
    )
    observed = list(checkpoint_revisions)
    if loaded_revision is not None:
        observed.append(loaded_revision)
    if revision is None or not observed or any(sha != revision for sha in observed):
        raise ValueError(
            f"loaded model revision mismatch or unverifiable: "
            f"expected {revision!r}, observed {observed!r}"
        )
    return revision


def _verify_saved_revision(index, model_name, revision):
    metadata = json.loads((index / "metadata.json").read_text(encoding="utf-8"))
    saved = metadata["embedding_config"]
    if saved.get("model_name") != model_name:
        raise ValueError("saved model differs from requested model")
    if "revision" not in saved or saved["revision"] != revision:
        raise ValueError("saved metadata revision differs from requested revision")


def _runtime_threads() -> dict[str, int]:
    import faiss
    import torch

    return {
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "faiss_omp_max_threads": faiss.omp_get_max_threads(),
    }


def _code_provenance() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "data/loader.py",
        "data/preprocess.py",
        "evaluation/constants.py",
        "evaluation/dense_runtime.py",
        "evaluation/json_io.py",
        "evaluation/latency.py",
        "scripts/benchmark_dense_runtime.py",
        "retrievers/dense.py",
        "retrievers/model_config.py",
        "config/dense_models.toml",
        "indexing/faiss_index.py",
    )
    result = {
        "sha256": {
            name: _sha256(root / name) if (root / name).is_file() else None
            for name in paths
        }
    }
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        result["working_tree_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
        result["working_tree_dirty"] = None
    return result


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()

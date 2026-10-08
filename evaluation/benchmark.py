"""공통 dataset의 검색 품질과 warm-up 이후 query-time latency를 비교합니다."""

import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from math import isfinite
from numbers import Real
from pathlib import Path
from time import perf_counter

from evaluation.data import EvaluationDataset, file_sha256, load_evaluation_dataset
from evaluation.latency import (
    DEFAULT_REPEATS,
    DEFAULT_WARMUP,
    latency_summary,
    measure_query_latency,
)
from evaluation.metrics import (
    EVALUATION_TOP_K,
    METRIC_NAMES,
    evaluate_query,
    evaluate_run,
)
from evaluation.runtime_config import CUDA_DEVICE, THREAD_ENV
from fusion.rrf import DEFAULT_RANK_CONSTANT
from retrievers.base import Retriever, SearchResult

METHODS = ("tfidf", "bm25", "dense", "hybrid")
SUMMARY_FIELDS = (
    "method",
    *METRIC_NAMES,
    "latency_mean_ms",
    "latency_p95_ms",
    "setup_seconds",
    "warmup_seconds",
    "query_count",
    "sample_count",
)


def benchmark_retriever(
    retriever: Retriever,
    dataset: EvaluationDataset,
    *,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
) -> dict:
    """검색기를 재사용하고 각 pass에서 동일한 순서로 모든 query를 검색합니다."""
    _positive(warmup, "warmup")
    _positive(repeats, "repeats")
    if not dataset.queries:
        raise ValueError("평가할 질의가 하나 이상 필요합니다.")
    query_ids = [query.query_id for query in dataset.queries]
    if len(set(query_ids)) != len(query_ids) or set(query_ids) != set(dataset.qrels):
        raise ValueError("queries와 qrels의 query_id 집합이 일치해야 합니다.")
    corpus_ids = {document.document_id for document in dataset.documents}

    def _validate_results(position, results):
        _check_results(results, dataset.qrels[query_ids[position]], corpus_ids)

    measurement = measure_query_latency(
        retriever,
        [query.text for query in dataset.queries],
        top_k=EVALUATION_TOP_K,
        warmup=warmup,
        repeats=repeats,
        validate_results=_validate_results,
    )
    run = dict(zip(query_ids, measurement.first_results, strict=True))
    samples = dict(zip(query_ids, measurement.samples_ms, strict=True))

    queries = []
    for query in dataset.queries:
        query_id = query.query_id
        queries.append(
            {
                "query_id": query_id,
                "text": query.text,
                "metrics": evaluate_query(run[query_id], dataset.qrels[query_id]),
                "latency": latency_summary(samples[query_id]),
                "latency_samples_ms": samples[query_id],
                "results": [asdict(result) for result in run[query_id]],
            }
        )
    return {
        "metrics": evaluate_run(run, dataset.qrels),
        "latency": measurement.latency,
        "warmup_seconds": measurement.warmup_seconds,
        "query_count": len(queries),
        "queries": queries,
    }


def run_benchmark(
    data_dir: str | Path,
    *,
    split: str = "dev",
    methods: tuple[str, ...] = ("tfidf", "bm25"),
    dense_index: str | Path | None = None,
    device: str | None = None,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
    progress=None,
    strict_runtime: bool = False,
) -> dict:
    """같은 corpus와 query/qrels로 요청한 모든 방법을 평가하며 실패를 숨기지 않습니다."""
    _validate_methods(methods, dense_index)
    _positive(warmup, "warmup")
    _positive(repeats, "repeats")
    if device is not None and (not isinstance(device, str) or not device.strip()):
        raise ValueError("device는 비어 있지 않은 문자열이어야 합니다.")
    dataset = load_evaluation_dataset(data_dir, split)
    report = {
        "schema_version": "1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"split": split, **dataset.provenance},
        "conditions": {
            "top_k": EVALUATION_TOP_K,
            "warmup_passes": warmup,
            "measurement_passes": repeats,
            "query_order": [query.query_id for query in dataset.queries],
            "method_order": list(methods),
            "latency_unit": "ms",
            "latency_scope": "retriever.search only; includes query processing and result construction",
            "p95_method": "linear interpolation at (N - 1) * 0.95",
            "quality_run": "first measured pass",
            "relevance": "binary: relevance > 0",
            "reference_only": "Recall@100 is not included in this Top-10 run",
        },
        "environment": _environment(),
        "code": _code_provenance(),
        "methods": {},
    }
    retrievers = {}
    setup = {}
    configs = {}

    def _prepare(name):
        if name in retrievers:
            return retrievers[name]
        if progress:
            progress(f"검색기 준비: {name}")
        started = perf_counter()
        retriever, config = _build_retriever(name, dataset, dense_index, device)
        # Dense 모델은 lazy load이므로 첫 실제 검색도 준비 시간에 포함합니다.
        if name == "dense" and strict_runtime:
            from evaluation.dense_runtime import (
                _checkpoint_revisions,
                _verify_loaded_revision,
            )

            capture = _checkpoint_revisions()
        else:
            capture = nullcontext([])
        with capture as observed:
            results = retriever.search(dataset.queries[0].text, EVALUATION_TOP_K)
        if name == "dense" and strict_runtime:
            config["loaded_model_revision"] = _verify_loaded_revision(
                retriever, retriever.config.revision, observed
            )
            config["runtime_dtype"] = str(
                next(retriever.embedder.model.parameters()).dtype
            )
            config["max_seq_length"] = retriever.embedder.model.max_seq_length
            if (
                config["runtime_dtype"] != "torch.float32"
                or str(retriever.embedder.model.device) != CUDA_DEVICE
            ):
                raise ValueError("strict experiment requires FP32 on cuda:0")
        _check_results(
            results,
            dataset.qrels[dataset.queries[0].query_id],
            {doc.document_id for doc in dataset.documents},
        )
        setup[name] = perf_counter() - started
        if name == "dense":
            config["runtime_device"] = str(
                getattr(retriever.embedder.model, "device", "unknown")
            )
        retrievers[name] = retriever
        configs[name] = config
        return retriever

    for method in methods:
        if method == "hybrid":
            from fusion.hybrid import HybridRetriever

            sparse, dense = _prepare("bm25"), _prepare("dense")
            started = perf_counter()
            retriever = HybridRetriever(
                sparse, dense, rank_constant=DEFAULT_RANK_CONSTANT
            )
            setup[method] = setup["bm25"] + setup["dense"] + perf_counter() - started
            config = {
                "sparse": "bm25",
                "dense": configs["dense"],
                "rank_constant": DEFAULT_RANK_CONSTANT,
                "candidate_top_k_per_component": EVALUATION_TOP_K,
            }
        else:
            retriever = _prepare(method)
            config = configs[method]
        if progress:
            progress(
                f"평가: {method} / {len(dataset.queries)} queries / {repeats} repeats"
            )
        report["methods"][method] = {
            "config": config,
            "setup_seconds": setup[method],
            **benchmark_retriever(retriever, dataset, warmup=warmup, repeats=repeats),
        }
    return report


def check_output_directory(directory: str | Path) -> None:
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("output-dir은 새 경로 또는 비어 있는 디렉터리여야 합니다.")


def write_report(report: dict, directory: str | Path) -> None:
    """JSON, 요약/질의/원시 latency CSV와 비교표를 UTF-8로 저장합니다."""
    directory = Path(directory)
    check_output_directory(directory)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
        stage = Path(temporary)
        (stage / "results.json").write_text(payload, encoding="utf-8")
        summary = []
        query_rows = []
        samples = []
        for method, result in report["methods"].items():
            summary.append(
                {
                    "method": method,
                    **result["metrics"],
                    "latency_mean_ms": result["latency"]["mean_ms"],
                    "latency_p95_ms": result["latency"]["p95_ms"],
                    "setup_seconds": result["setup_seconds"],
                    "warmup_seconds": result["warmup_seconds"],
                    "query_count": result["query_count"],
                    "sample_count": result["latency"]["sample_count"],
                }
            )
            for query in result["queries"]:
                query_rows.append(
                    {
                        "method": method,
                        "query_id": query["query_id"],
                        "text": query["text"],
                        **query["metrics"],
                        "latency_mean_ms": query["latency"]["mean_ms"],
                        "latency_p95_ms": query["latency"]["p95_ms"],
                        "result_count": len(query["results"]),
                    }
                )
                samples.extend(
                    {
                        "method": method,
                        "query_id": query["query_id"],
                        "repetition": number,
                        "latency_ms": value,
                    }
                    for number, value in enumerate(query["latency_samples_ms"], start=1)
                )
        _write_csv(stage / "summary.csv", SUMMARY_FIELDS, summary)
        _write_csv(
            stage / "queries.csv",
            (
                "method",
                "query_id",
                "text",
                "recall@5",
                "recall@10",
                "rr@10",
                "ndcg@10",
                "latency_mean_ms",
                "latency_p95_ms",
                "result_count",
            ),
            query_rows,
        )
        _write_csv(
            stage / "latencies.csv",
            ("method", "query_id", "repetition", "latency_ms"),
            samples,
        )
        table = [
            "# Retrieval Benchmark",
            "",
            f"Split: {report['dataset']['split']} · corpus: {report['dataset']['corpus_size']} · queries: {report['dataset']['query_count']} · Top-K: 10",
            "",
            "| Method | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean ms | P95 ms |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in summary:
            table.append(
                "| "
                + row["method"]
                + " | "
                + " | ".join(
                    f"{row[name]:.6f}"
                    for name in (*METRIC_NAMES, "latency_mean_ms", "latency_p95_ms")
                )
                + " |"
            )
        table.extend(
            [
                "",
                "Query-time latency만 비교합니다. 준비 시간과 warm-up은 포함하지 않습니다.",
                "Recall@100은 참고용이며 이 공통 Top-10 결과표에 포함하지 않습니다.",
                "10k subset은 전체 MIRACL 공식 benchmark가 아닙니다.",
                "",
            ]
        )
        (stage / "comparison.md").write_text("\n".join(table), encoding="utf-8")
        directory.mkdir(exist_ok=True)
        for path in stage.iterdir():
            path.replace(directory / path.name)


def _build_retriever(name, dataset, dense_index, device):
    if name == "tfidf":
        from retrievers.tfidf import TfidfRetriever

        return TfidfRetriever(dataset.documents), {
            "analyzer": "char_wb",
            "ngram_range": [2, 4],
            "norm": "l2",
        }
    if name == "bm25":
        from retrievers.bm25 import BM25Retriever

        return BM25Retriever(dataset.documents), {
            "analyzer": "char_wb",
            "ngram_range": [2, 4],
            "method": "lucene",
            "idf_method": "lucene",
            "k1": 1.5,
            "b": 0.75,
        }
    if name == "dense":
        index = Path(dense_index)
        for filename in ("metadata.json", "index.faiss"):
            if not (index / filename).is_file():
                raise ValueError(f"Dense 인덱스 파일이 없습니다: {index / filename}")
        from retrievers.dense import DenseRetriever

        retriever = DenseRetriever.load(index, device=device)
        if tuple(retriever.documents) != dataset.documents:
            raise ValueError(
                "Dense 인덱스의 문서 ID / 순서 / 원문이 공통 corpus와 다릅니다."
            )
        return retriever, {
            "index": str(index),
            "metadata_sha256": file_sha256(index / "metadata.json"),
            "index_sha256": file_sha256(index / "index.faiss"),
            "embedding": asdict(retriever.config),
        }
    raise ValueError(f"지원하지 않는 검색 방식: {name}")


def _write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _check_results(results: list[SearchResult], qrels, corpus_ids):
    if not isinstance(results, list) or len(results) > EVALUATION_TOP_K:
        raise ValueError("search는 최대 Top-10의 SearchResult 목록을 반환해야 합니다.")
    evaluate_query(results, qrels)
    for result in results:
        if result.document_id not in corpus_ids:
            raise ValueError(f"검색 결과 ID가 corpus에 없습니다: {result.document_id}")
        if (
            isinstance(result.score, bool)
            or not isinstance(result.score, Real)
            or not isfinite(result.score)
        ):
            raise ValueError("검색 결과 score는 유한한 숫자여야 합니다.")


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name}는 양의 정수여야 합니다.")


def _validate_methods(methods, dense_index):
    if (
        not methods
        or len(set(methods)) != len(methods)
        or any(method not in METHODS for method in methods)
    ):
        raise ValueError(
            "methods는 중복 없는 tfidf / bm25 / dense / hybrid 목록이어야 합니다."
        )
    if set(methods) & {"dense", "hybrid"} and dense_index is None:
        raise ValueError(
            "Dense 또는 Hybrid 평가에는 --index로 공통 corpus의 저장된 Dense 인덱스를 지정해야 합니다."
        )


def _environment():
    packages = {}
    for name in (
        "scikit-learn",
        "bm25s",
        "numpy",
        "scipy",
        "sentence-transformers",
        "faiss-cpu",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (*THREAD_ENV, "TOKENIZERS_PARALLELISM")
        },
    }


def _code_provenance():
    root = Path(__file__).resolve().parents[1]
    paths = (
        "evaluation/data.py",
        "evaluation/metrics.py",
        "evaluation/latency.py",
        "evaluation/benchmark.py",
        "scripts/evaluate.py",
        "retrievers/tfidf.py",
        "retrievers/bm25.py",
        "retrievers/dense.py",
        "retrievers/model_config.py",
        "config/dense_models.toml",
        "indexing/faiss_index.py",
        "fusion/hybrid.py",
        "fusion/rrf.py",
        "data/loader.py",
        "data/preprocess.py",
    )
    result = {
        "sha256": {
            name: file_sha256(root / name) for name in paths if (root / name).is_file()
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

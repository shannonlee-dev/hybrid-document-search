"""Measure live BGE-M3 embedding through Top-10 on the pinned Ko-MIRACL Dev50.

Model/index initialization and HTTP are excluded. The cached-vector benchmark
remains separate. Run from the repository root in a dedicated process.
"""

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter
from types import SimpleNamespace

from evaluation.constants import CUDA_DEVICE, DEFAULT_THREADS, MILLISECONDS_PER_SECOND
from evaluation.constants import EVALUATION_TOP_K as _TOP_K
from evaluation.dense_runtime import _checkpoint_revisions, _verify_loaded_revision
from evaluation.json_io import write_json
from evaluation.latency import latency_summary
from evaluation.metrics import evaluate_run
from evaluation.runtime_config import configure_runtime
from fusion.hybrid import HybridRetriever
from fusion.score import DEFAULT_BM25_WEIGHT, ScoreFusionRetriever
from indexing.faiss_index import FaissIndex
from retrievers.bm25 import BM25Retriever
from retrievers.dense import DenseConfig, DenseEmbedder, DenseRetriever
from scripts._cli import _positive_int
from scripts.benchmark_score_fusion import _load_cache, _sha256

_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE = _ROOT / "docs/experiments/score-fusion/miracl-dev.json"
_ENCODING = _ROOT / "docs/experiments/score-fusion/encoding/miracl10k.json"
_DEFAULT_WARMUP_REQUESTS = 10
_DEFAULT_REPEATS = 10
_EXPECTED_CORPUS_SIZE = 10000
_EXPECTED_QUERY_COUNT = 50


class _PreparedQuery:
    """Pass this request's freshly inferred vector into unchanged retrievers once."""

    def __init__(self, config):
        self.config = config
        self._query = None
        self._vector = None

    def _set(self, query, vector):
        self._query, self._vector = query, vector

    def encode_query(self, query):
        if self._vector is None or query != self._query:
            raise ValueError("missing, stale or mismatched live query embedding")
        vector, self._vector = self._vector, None
        return vector


def _build_methods(documents, vectors, prepared):
    sparse = BM25Retriever(documents)
    dense = DenseRetriever(
        index=FaissIndex.build(vectors), documents=documents, embedder=prepared
    )
    return {
        "dense": dense,
        "rrf": HybridRetriever(sparse, dense),
        f"score_{DEFAULT_BM25_WEIGHT:g}": ScoreFusionRetriever(
            sparse, dense, bm25_weight=DEFAULT_BM25_WEIGHT
        ),
    }


def _measure(
    methods,
    embedder,
    prepared,
    queries,
    query_ids,
    synchronize,
    *,
    warmup=_DEFAULT_WARMUP_REQUESTS,
    repeats=_DEFAULT_REPEATS,
):
    """Rotate methods; encode anew for every warm-up and measured request."""
    names = list(methods)
    samples = {name: [] for name in names}
    runs = {name: {} for name in names}

    def _request(name, query_id):
        query = queries[query_id]
        synchronize()  # 이전 GPU 작업의 대기 시간은 측정에서 제외한다.
        started = perf_counter()
        vector = embedder.encode_query(query)
        synchronize()  # 추론 완료와 CPU 전송까지 임베딩 시간에 포함한다.
        embedded = perf_counter()
        prepared._set(query, vector)
        results = methods[name].search(query, _TOP_K)
        finished = perf_counter()  # 이후 검색과 융합은 CPU에서 동기적으로 완료된다.
        return results, {
            "ms": (finished - started) * MILLISECONDS_PER_SECOND,
            "embedding_ms": (embedded - started) * MILLISECONDS_PER_SECOND,
            "retrieval_fusion_ms": (finished - embedded) * MILLISECONDS_PER_SECOND,
        }

    # 고정된 질의 순서로 순환하며 방식별로 지정된 횟수만큼 준비 요청을 실행한다.
    for step in range(warmup):
        offset = step % len(names)
        for name in names[offset:] + names[:offset]:
            _request(name, query_ids[step % len(query_ids)])
    for repetition in range(repeats):
        for index, query_id in enumerate(query_ids):
            offset = (index + repetition) % len(names)
            for position, name in enumerate(names[offset:] + names[:offset]):
                results, row = _request(name, query_id)
                # 순위 검증과 보고서 기록 비용은 측정에서 제외한다.
                ids = [r.document_id for r in results]
                if len(ids) != len(set(ids)) or not ids:
                    raise ValueError(f"invalid results: {name}/{query_id}")
                if query_id in runs[name]:
                    if ids != [r.document_id for r in runs[name][query_id]]:
                        raise ValueError(f"unstable ranking: {name}/{query_id}")
                else:
                    runs[name][query_id] = results
                samples[name].append(
                    dict(
                        query_id=query_id,
                        repetition=repetition,
                        position=position,
                        **row,
                    )
                )
        print(f"E2E repetition {repetition + 1}/{repeats}", file=sys.stderr, flush=True)
    return samples, runs


def _summary(values):
    return {**latency_summary(values), "median_ms": median(values)}


def _run(args):
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    data, documents, vectors, _, query_ids, hashes, encoding = _load_cache(
        args.cache_dir, "dev", _REFERENCE
    )
    if (
        len(query_ids) != _EXPECTED_QUERY_COUNT
        or len(documents) != _EXPECTED_CORPUS_SIZE
    ):
        raise ValueError("requires the pinned Ko-MIRACL Dev50 / 10k corpus")
    if _sha256(args.cache_dir / "encoding.json") != _sha256(_ENCODING):
        raise ValueError("encoding record differs from pinned BGE-M3 provenance")
    config = DenseConfig(
        **{
            **encoding["model"],
            "device": args.device,
            "batch_size": 1,
        }
    )
    runtime = configure_runtime(args.threads, cuda=args.device == CUDA_DEVICE)
    import torch
    from threadpoolctl import threadpool_info

    synchronize = (
        (lambda: torch.cuda.synchronize(args.device))
        if args.device == CUDA_DEVICE
        else lambda: None
    )
    embedder = DenseEmbedder(config)
    # 최초 모델 로딩과 인코딩은 준비 요청 및 측정 샘플에서 제외한다.
    with _checkpoint_revisions() as revisions:
        embedder.encode_query(data["queries"][query_ids[0]])
    loaded_revision = _verify_loaded_revision(
        SimpleNamespace(config=config, embedder=embedder), config.revision, revisions
    )
    model = embedder.model
    dtype = str(next(model.parameters()).dtype)
    if dtype != encoding["dtype"] or model.max_seq_length != encoding["max_seq_length"]:
        raise ValueError("live model precision/max length differs from corpus encoding")
    if str(model.device) != args.device:
        raise ValueError("loaded device differs from requested device")
    prepared = _PreparedQuery(config)
    methods = _build_methods(documents, vectors, prepared)
    samples, runs = _measure(
        methods,
        embedder,
        prepared,
        data["queries"],
        query_ids,
        synchronize,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    qrels = {qid: data["qrels"][qid] for qid in query_ids}
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": data["name"],
        "split": "dev",
        "corpus_size": len(documents),
        "query_count": len(query_ids),
        "query_ids": query_ids,
        "top_k": _TOP_K,
        "bm25_weight": DEFAULT_BM25_WEIGHT,
        "latency_scope": "live query embedding start through final Top-10 return",
        "latency_excludes": ["model/index initialization", "HTTP", "serialization"],
        "latency": {
            name: _summary([r["ms"] for r in rows]) for name, rows in samples.items()
        },
        "stages": {
            name: {
                stage: _summary([r[stage] for r in rows])
                for stage in ("embedding_ms", "retrieval_fusion_ms")
            }
            for name, rows in samples.items()
        },
        "timing": {
            "warmup_requests_per_method": args.warmup,
            "repeats_per_query": args.repeats,
            "queries": len(query_ids),
            "order": "rotating methods by (query index + repetition) modulo 3",
            "warmup_order": "rotating methods; cycle through fixed query order",
            "aggregation": "pooled request samples; linearly interpolated P95",
            "cuda_synchronization": "before start and after embedding"
            if args.device == CUDA_DEVICE
            else None,
            "embedding": "fresh per method/request; tokenization, inference, normalization, CPU transfer",
            "retrieval_fusion": "prepared-vector handoff, CPU search/fusion/sort and SearchResult construction",
            "rrf_order": "embedding first, then BM25 and Dense Top-10, k=60 RRF",
            "score_path": "NumPy full-corpus inner product + BM25, corpus-max normalization",
            "dense_path": "FAISS CPU IndexFlatIP",
        },
        "quality": {name: evaluate_run(run, qrels) for name, run in runs.items()},
        "input_sha256": hashes,
        "reference_summary_sha256": _sha256(_REFERENCE),
        "encoding_sha256": _sha256(_ENCODING),
        "model_config": asdict(config),
        "loaded_revision": loaded_revision,
        "dtype": dtype,
        "max_seq_length": model.max_seq_length,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                platform.processor(),
            ),
            "runtime": runtime,
            "threadpools_after_run": threadpool_info(),
            "packages": {
                name: version(name)
                for name in (
                    "torch",
                    "sentence-transformers",
                    "transformers",
                    "numpy",
                    "faiss-cpu",
                    "bm25s",
                    "scikit-learn",
                )
            },
        },
        "code_sha256": {
            str(path.relative_to(_ROOT)): _sha256(path)
            for path in (
                Path(__file__).resolve(),
                *(
                    _ROOT / name
                    for name in (
                        "scripts/benchmark_score_fusion.py",
                        "retrievers/dense.py",
                        "retrievers/bm25.py",
                        "fusion/hybrid.py",
                        "fusion/rrf.py",
                        "fusion/score.py",
                        "indexing/faiss_index.py",
                        "indexing/vectors.py",
                        "evaluation/runtime_config.py",
                        "evaluation/constants.py",
                        "data/preprocess.py",
                        "retrievers/base.py",
                        "evaluation/latency.py",
                        "evaluation/dense_runtime.py",
                        "evaluation/metrics.py",
                    )
                ),
            )
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "timings.json", samples)
    write_json(
        args.output / "runs.json",
        {
            name: {qid: [asdict(r) for r in results] for qid, results in run.items()}
            for name, run in runs.items()
        },
    )
    summary["artifact_sha256"] = {
        name: _sha256(args.output / name) for name in ("timings.json", "runs.json")
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary["latency"], indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_ROOT / "artifacts/score-fusion-cache/miracl10k",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=(CUDA_DEVICE, "cpu"), default=CUDA_DEVICE)
    parser.add_argument("--threads", type=_positive_int, default=DEFAULT_THREADS)
    parser.add_argument(
        "--warmup", type=_positive_int, default=_DEFAULT_WARMUP_REQUESTS
    )
    parser.add_argument("--repeats", type=_positive_int, default=_DEFAULT_REPEATS)
    args = parser.parse_args(argv)
    _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

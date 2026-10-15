"""Reproduce score-fusion quality and CPU retrieval latency from frozen vectors.

Input: data.json (documents, query_ids, queries, qrels, optional splits),
document-vectors.npy and query-vectors.npy in matching row order. No downloads
or model inference. Existing metric implementations and real BM25/FAISS are used.
"""

import argparse
import hashlib
import json
import platform
import random
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import faiss
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from data.loader import Document
from evaluation.constants import (
    DEFAULT_REPEATS,
    DEFAULT_THREADS,
    MILLISECONDS_PER_SECOND,
)
from evaluation.constants import EVALUATION_TOP_K as _TOP_K
from evaluation.json_io import read_json, write_json
from evaluation.latency import latency_summary
from evaluation.metrics import evaluate_query, evaluate_run
from fusion.hybrid import HybridRetriever
from fusion.rrf import reciprocal_rank_fusion
from fusion.score import DEFAULT_BM25_WEIGHT, ScoreFusionRetriever
from indexing.faiss_index import FaissIndex
from indexing.vectors import as_vectors
from retrievers.bm25 import DEFAULT_B, DEFAULT_K1, SCORING_METHOD, BM25Retriever
from retrievers.dense import DenseConfig, DenseRetriever
from scripts._cli import _positive_int

_ROOT = Path(__file__).resolve().parents[1]
_SEED = 20261014
_PCA_COMPONENTS = 8
_PCA_SHRINK = 0.25
_VECTOR_NORM_ATOL = 1e-4
_QUALITY_PROGRESS_INTERVAL = 100


class _PcaCorrection:
    """Center and shrink leading PCA directions for the historical ablation."""

    def __init__(self, documents):
        if len(documents) < 2:
            raise ValueError("PCA correction requires at least two documents")
        self._mean = documents.mean(axis=0)
        _, basis = np.linalg.eigh(np.atleast_2d(np.cov(documents, rowvar=False)))
        self._basis = basis[:, ::-1][
            :, : min(_PCA_COMPONENTS, documents.shape[1])
        ].astype(np.float32)

    def _transform(self, vectors):
        """Return corrected unit vectors without modifying the input array."""
        centered = vectors - self._mean
        corrected = centered - _PCA_SHRINK * (centered @ self._basis) @ self._basis.T
        corrected = as_vectors(corrected)
        norms = np.linalg.norm(corrected, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("PCA correction requires finite, nonzero vector norms")
        return corrected / norms


class _CachedEmbedder:
    """Provide cached query vectors through the Dense embedder interface."""

    def __init__(self, data, vectors, correction=None):
        # 캐시 어댑터의 설정만으로는 벡터를 생성한 모델을 검증할 수 없다.
        self.config = DenseConfig(model_name="cached-vectors")
        self._correction = correction
        self._vectors = {}
        for query_id, vector in zip(data["query_ids"], vectors, strict=True):
            text = data["queries"][query_id]
            if text in self._vectors and not np.array_equal(
                self._vectors[text], vector
            ):
                raise ValueError("duplicate query text has different cached vectors")
            self._vectors[text] = vector

    def encode_query(self, query):
        vector = self._vectors[query][None, :]
        return self._correction._transform(vector) if self._correction else vector


def _sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _load_cache(directory, split, expected_summary=None):
    """Load aligned unit vectors and validate query splits and recorded hashes."""
    hashes = {
        name: _sha256(directory / name)
        for name in ("data.json", "document-vectors.npy", "query-vectors.npy")
    }
    if expected_summary is not None:
        expected = read_json(expected_summary)["input_sha256"]
        for name, digest in hashes.items():
            if expected.get(name) != digest:
                raise ValueError(f"{name} checksum differs from expected summary")
    data = read_json(directory / "data.json")
    documents = [Document(**item) for item in data["documents"]]
    vectors = as_vectors(
        np.load(directory / "document-vectors.npy", allow_pickle=False)
    )
    queries = as_vectors(
        np.load(directory / "query-vectors.npy", allow_pickle=False), vectors.shape[1]
    )
    if len(vectors) != len(documents) or len(queries) != len(data["query_ids"]):
        raise ValueError("cache vector counts do not match document/query mappings")
    if len(set(data["query_ids"])) != len(data["query_ids"]):
        raise ValueError("duplicate query IDs")
    if not all(
        np.allclose(np.linalg.norm(array, axis=1), 1, atol=_VECTOR_NORM_ATOL)
        for array in (vectors, queries)
    ):
        raise ValueError("cache must contain normalized vectors")
    splits = data.get("splits", {})
    if set(splits.get("train", [])) & set(splits.get("dev", [])):
        raise ValueError("train and dev query IDs overlap")
    query_ids = data["splits"][split] if split else data["query_ids"]
    if not query_ids or len(set(query_ids)) != len(query_ids):
        raise ValueError("evaluation split must contain unique queries")
    if set(query_ids) - set(data["query_ids"]):
        raise ValueError("split contains unknown query IDs")
    encoding_path = directory / "encoding.json"
    encoding = read_json(encoding_path) if encoding_path.exists() else None
    if encoding:
        for role, filename in (
            ("document", "document-vectors.npy"),
            ("query", "query-vectors.npy"),
        ):
            if encoding["vector_sha256"][role] != hashes[filename]:
                raise ValueError(f"{role} vector checksum differs from encoding record")
    return data, documents, vectors, queries, query_ids, hashes, encoding


def _measure(methods, data, query_ids, runs, repeats, limit):
    """Time warmed retrieval paths and require the same ranking as quality runs."""
    chosen = random.Random(_SEED).sample(query_ids, min(limit, len(query_ids)))
    names = list(methods)
    samples = {name: [] for name in names}
    for query_id in chosen:
        for retriever in methods.values():
            retriever.search(data["queries"][query_id], _TOP_K)
    for repetition in range(repeats):
        for index, query_id in enumerate(chosen):
            offset = (index + repetition) % len(names)
            for name in names[offset:] + names[:offset]:
                started = perf_counter()
                results = methods[name].search(data["queries"][query_id], _TOP_K)
                elapsed = (perf_counter() - started) * MILLISECONDS_PER_SECOND
                expected = runs[name][query_id]
                if [r.document_id for r in results] != [
                    r.document_id for r in expected
                ]:
                    raise ValueError(
                        f"quality/runtime ranking mismatch: {name}/{query_id}"
                    )
                samples[name].append({"query_id": query_id, "ms": elapsed})
    return samples


def _verify_results(quality, runs, expected_summary, expected_runs):
    """Compare quality and optional ordered document IDs, never noisy latency."""
    checks = []
    if expected_summary is not None:
        expected = read_json(expected_summary).get("quality")
        if not expected or set(expected) != set(quality):
            raise ValueError("expected quality must contain the same methods")
        for name, metrics in quality.items():
            if set(metrics) != set(expected[name]) or any(
                not np.isclose(value, expected[name][metric], rtol=1e-9, atol=1e-12)
                for metric, value in metrics.items()
            ):
                raise ValueError(f"quality differs from expected summary: {name}")
        checks.extend(["input_sha256", "quality"])
    if expected_runs is not None:
        expected = read_json(expected_runs)
        actual_ids = {
            name: {qid: [hit.document_id for hit in hits] for qid, hits in run.items()}
            for name, run in runs.items()
        }
        expected_ids = {
            name: {
                qid: [hit["document_id"] for hit in hits] for qid, hits in run.items()
            }
            for name, run in expected.items()
        }
        if actual_ids != expected_ids:
            raise ValueError("ranking differs from expected runs")
        checks.append("rankings")
    return checks


def _run(args):
    """Evaluate cached vectors and write reports to a new output directory."""
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    data, documents, vectors, queries, query_ids, hashes, encoding = _load_cache(
        args.cache_dir, args.split, args.expected_summary
    )
    sparse = BM25Retriever(documents)
    dense = DenseRetriever(
        index=FaissIndex.build(vectors),
        documents=documents,
        embedder=_CachedEmbedder(data, queries),
    )
    methods = {"dense": dense, "rrf": HybridRetriever(sparse, dense)}
    score = ScoreFusionRetriever(sparse, dense)
    for weight in args.weights:
        methods[f"score_{weight:g}"] = score.with_bm25_weight(weight)
    corrected = None
    if args.correction:
        correction = _PcaCorrection(vectors)
        corrected = DenseRetriever(
            index=FaissIndex.build(correction._transform(vectors)),
            documents=documents,
            embedder=_CachedEmbedder(data, queries, correction),
        )
        methods["corrected_score"] = ScoreFusionRetriever(sparse, corrected)
    runs = {name: {} for name in [*methods, "rrf_full"]}
    for position, query_id in enumerate(query_ids):
        query = data["queries"][query_id]
        dense_all = dense.search(query, len(documents))
        sparse_all = sparse.search(query, len(documents))
        # FAISS의 동점 순서는 k에 따라 달라질 수 있어 실제 top-k 경로를 호출합니다.
        runs["dense"][query_id] = dense.search(query, _TOP_K)
        runs["rrf"][query_id] = methods["rrf"].search(query, _TOP_K)
        runs["rrf_full"][query_id] = reciprocal_rank_fusion(
            [sparse_all, dense_all], _TOP_K
        )
        for weight in args.weights:
            runs[f"score_{weight:g}"][query_id] = methods[f"score_{weight:g}"].search(
                query, _TOP_K
            )
        if corrected:
            runs["corrected_score"][query_id] = methods["corrected_score"].search(
                query, _TOP_K
            )
        if (position + 1) % _QUALITY_PROGRESS_INTERVAL == 0:
            print(f"quality {position + 1}/{len(query_ids)}", file=sys.stderr)
    qrels = {query_id: data["qrels"][query_id] for query_id in query_ids}
    quality = {name: evaluate_run(run, qrels) for name, run in runs.items()}
    verified = _verify_results(quality, runs, args.expected_summary, args.expected_runs)
    samples = _measure(
        methods, data, query_ids, runs, args.repeats, args.timing_queries
    )
    per_query = {
        name: {qid: evaluate_query(results, qrels[qid]) for qid, results in run.items()}
        for name, run in runs.items()
    }
    summary = {
        "schema_version": 3,
        "verified_against_reference": verified,
        "expected_runs_sha256": _sha256(args.expected_runs)
        if args.expected_runs
        else None,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": data["name"],
        "split": args.split,
        "corpus_size": len(documents),
        "query_count": len(query_ids),
        "weights": args.weights,
        "top_k": _TOP_K,
        "quality": quality,
        "latency": {
            n: latency_summary([s["ms"] for s in ss]) for n, ss in samples.items()
        },
        "latency_excludes": ["model_inference", "initialization", "HTTP"],
        "latency_scope": "CPU retrieval and fusion from cached query vectors",
        "latency_details": {
            "includes": [
                "cached query lookup",
                "retrieval",
                "fusion/sorting",
                "SearchResult construction",
            ],
            "excludes": ["API serialization", "concurrent load"],
            "dense_path": "FAISS IndexFlatIP",
            "score_path": "NumPy full-corpus inner product + BM25",
            "rrf_full": "quality only; no latency measurement",
        },
        "timing": {
            "seed": _SEED,
            "warmup": 1,
            "repeats": args.repeats,
            "queries": min(args.timing_queries, len(query_ids)),
            "order": "rotating methods per query/repetition",
        },
        "candidates": {
            "rrf": f"each source top {_TOP_K}",
            "rrf_full": "full corpus",
            "score": "full corpus",
            "bm25_normalization": "corpus maximum",
        },
        "bm25": {
            "analyzer": "char_wb",
            "ngram_range": [2, 4],
            "k1": DEFAULT_K1,
            "b": DEFAULT_B,
            "method": SCORING_METHOD,
        },
        "correction": {
            "center": True,
            "components": min(_PCA_COMPONENTS, vectors.shape[1]),
            "shrink": _PCA_SHRINK,
        }
        if args.correction
        else None,
        "input_directory": str(args.cache_dir.resolve()),
        "input_sha256": hashes,
        "encoding_record": encoding,
        "model_config": encoding.get("model") if encoding else None,
        "provenance_status": "encoding_record" if encoding else "unverified",
        "expected_summary_sha256": _sha256(args.expected_summary)
        if args.expected_summary
        else None,
        "code_sha256": {
            name: _sha256(_ROOT / name)
            for name in (
                "scripts/benchmark_score_fusion.py",
                "fusion/score.py",
                "fusion/hybrid.py",
                "fusion/rrf.py",
                "retrievers/bm25.py",
                "retrievers/dense.py",
                "retrievers/base.py",
                "retrievers/model_config.py",
                "indexing/faiss_index.py",
                "indexing/vectors.py",
                "data/loader.py",
                "data/preprocess.py",
                "evaluation/metrics.py",
                "evaluation/constants.py",
                "evaluation/latency.py",
                "evaluation/json_io.py",
            )
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in ("numpy", "faiss-cpu", "bm25s", "scikit-learn")
            },
            "threadpools": threadpool_info(),
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "runs.json",
        {
            name: {
                qid: [asdict(result) for result in results]
                for qid, results in run.items()
            }
            for name, run in runs.items()
        },
    )
    write_json(args.output / "per-query.json", per_query)
    write_json(args.output / "timings.json", samples)
    summary["artifact_sha256"] = {
        name: _sha256(args.output / name)
        for name in ("runs.json", "per-query.json", "timings.json")
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"quality": quality, "latency": summary["latency"]}, indent=2))


def main(argv=None):
    """Run the offline benchmark with the requested CPU thread limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-summary",
        type=Path,
        help="Verify cache input SHA-256 values and quality against a saved summary",
    )
    parser.add_argument(
        "--expected-runs",
        type=Path,
        help="Compare every method/query's ordered document IDs with a saved runs.json",
    )
    parser.add_argument(
        "--split", help="Existing split name; omit for all cached queries"
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, default=[DEFAULT_BM25_WEIGHT]
    )
    parser.add_argument("--correction", action="store_true")
    parser.add_argument("--timing-queries", type=_positive_int, default=25)
    parser.add_argument("--repeats", type=_positive_int, default=DEFAULT_REPEATS)
    parser.add_argument("--threads", type=_positive_int, default=DEFAULT_THREADS)
    args = parser.parse_args(argv)
    if any(not np.isfinite(w) or w < 0 for w in args.weights):
        parser.error("weights must be finite and non-negative")
    if len({f"score_{w:g}" for w in args.weights}) != len(args.weights):
        parser.error("weights must have distinct report names (six significant digits)")
    faiss.omp_set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

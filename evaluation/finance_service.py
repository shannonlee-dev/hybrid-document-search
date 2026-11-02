"""Evaluate the Finance query encoder through the search service path.

Conditions follow docs/validation/2026-11-02/finance-service-eval/preregistration.md.
Both encoders share frozen original BGE-M3 document vectors and the service
BM25/FAISS retrievers; only the query encoder differs. Nothing here trains,
selects a checkpoint, or tunes alpha.
"""

import hashlib
import json
import os
import time
from math import fsum
from pathlib import Path

from app.schemas import RetrievalMethod
from app.service import SearchService, build_search_service
from data.loader import Document
from evaluation.data import file_sha256
from evaluation.json_io import read_json
from evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from retrievers.base import SearchResult

CHECKPOINT_SHA256 = "fbbd5ed0c4d82c42745e21afb54c9a43000d38d39a975595b61b52f314e42971"
METRICS = ("p@1", "recall@5", "recall@10", "mrr@10", "ndcg@10")
TOP_K = 10
SHARED_ALPHA = 0.84
FINANCE_SELECTED_ALPHA = 0.89
BOOTSTRAP_SEED = 20261102
BOOTSTRAP_RESAMPLES = 20000
CONDITIONS = {
    # id: (encoder, kind, alpha, role)
    "A-bge": ("bge", "service_hybrid", None, "co-primary A"),
    "A-fin": ("fin", "service_hybrid", None, "co-primary A"),
    "B-bge": ("bge", "fixed", SHARED_ALPHA, "co-primary B"),
    "B-fin": ("fin", "fixed", SHARED_ALPHA, "co-primary B"),
    "S-fin89": ("fin", "fixed", FINANCE_SELECTED_ALPHA, "secondary"),
    "D-bge": ("bge", "dense", None, "secondary"),
    "D-fin": ("fin", "dense", None, "secondary"),
}
PAIRS = {
    "A": ("A-fin", "A-bge"),
    "B": ("B-fin", "B-bge"),
    "S89-vs-B-bge": ("S-fin89", "B-bge"),
    "D": ("D-fin", "D-bge"),
}


# ------------------------------------------------------------------ helpers
def query_metrics(document_ids, judgments):
    hits = [SearchResult(doc, rank, 0.0) for rank, doc in enumerate(document_ids, 1)]
    return {
        "p@1": precision_at_k(hits, judgments, 1),
        "recall@5": recall_at_k(hits, judgments, 5),
        "recall@10": recall_at_k(hits, judgments, 10),
        "mrr@10": reciprocal_rank_at_k(hits, judgments, 10),
        "ndcg@10": ndcg_at_k(hits, judgments, 10),
    }


def write_corpus(documents, path):
    """Write prepared-format JSONL that build_search_service can load."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for doc in documents:
            stream.write(
                json.dumps(
                    {
                        "document_id": doc.document_id,
                        "text": doc.text,
                        "title": doc.title,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_index_dir(documents, vectors, directory):
    """Save frozen document vectors in the service's DenseRetriever format."""
    from indexing.faiss_index import FaissIndex
    from retrievers.dense import DenseConfig, DenseEmbedder, DenseRetriever

    directory = Path(directory)
    if directory.exists():
        raise ValueError(f"index directory already exists: {directory}")
    directory.mkdir(parents=True)
    retriever = DenseRetriever(
        index=FaissIndex.build(vectors),
        documents=documents,
        embedder=DenseEmbedder(DenseConfig()),
    )
    retriever.save(directory)
    return {
        "index_sha256": read_json(directory / "metadata.json")["index_sha256"],
        "documents": len(documents),
    }


def build_services(corpus_path, index_dir, device, checkpoint):
    """Build the original and Finance services exactly as the API lifespan does."""
    services, seconds = {}, {}
    for name, query_checkpoint in (("bge", None), ("fin", checkpoint)):
        started = time.perf_counter()
        service = build_search_service(
            Path(corpus_path),
            dense_index_path=Path(index_dir),
            dense_device=device,
            dense_query_checkpoint=query_checkpoint,
        )
        seconds[name] = time.perf_counter() - started
        for method in (
            RetrievalMethod.BM25,
            RetrievalMethod.DENSE,
            RetrievalMethod.HYBRID,
        ):
            if service.retriever(method) is None:
                raise RuntimeError(
                    f"{name} {method.value} unavailable: {service.availability()[method]}"
                )
        services[name] = service
    return services, seconds


def condition_services(services):
    """Map each condition to a SearchService and the API method it is served under."""
    from fusion.minmax import UnionMinMaxFusionRetriever

    mapped = {}
    for cid, (encoder, kind, alpha, _) in CONDITIONS.items():
        service = services[encoder]
        if kind == "service_hybrid":
            mapped[cid] = (service, RetrievalMethod.HYBRID)
        elif kind == "dense":
            mapped[cid] = (service, RetrievalMethod.DENSE)
        else:
            fusion = UnionMinMaxFusionRetriever(
                service.retriever(RetrievalMethod.BM25),
                service.retriever(RetrievalMethod.DENSE),
                alpha=alpha,
            )
            # Fixed fusion is served in the hybrid slot; the API schema is unchanged.
            mapped[cid] = (
                SearchService(
                    {
                        RetrievalMethod.BM25: service.retriever(RetrievalMethod.BM25),
                        RetrievalMethod.DENSE: service.retriever(RetrievalMethod.DENSE),
                        RetrievalMethod.HYBRID: fusion,
                    }
                ),
                RetrievalMethod.HYBRID,
            )
    return mapped


def run_conditions(mapped, queries, qrels, conditions=None):
    """Query every condition through POST /search and directly; return rows."""
    from fastapi.testclient import TestClient

    from app.api import create_app

    rows, mismatches = [], {}
    for cid in conditions or mapped:
        service, method = mapped[cid]
        mismatches[cid] = 0
        with TestClient(create_app(service=service)) as client:
            for query_id, text in queries:
                response = client.post(
                    "/search",
                    json={"query": text, "method": method.value, "top_k": TOP_K},
                )
                response.raise_for_status()
                api_ids = [hit["document_id"] for hit in response.json()["results"]]
                direct_ids = [
                    r.document_id for r in service.search(text, method, TOP_K)
                ]
                if api_ids != direct_ids:
                    mismatches[cid] += 1
                judgments = qrels[query_id]
                rows.append(
                    {
                        "condition": cid,
                        "query_id": query_id,
                        **query_metrics(api_ids, judgments),
                        "top10_document_ids": api_ids,
                        "relevant_rank_top10": next(
                            (
                                i
                                for i, d in enumerate(api_ids, 1)
                                if judgments.get(d, 0) > 0
                            ),
                            None,
                        ),
                    }
                )
    return rows, mismatches


def aggregate(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["condition"], []).append(row)
    return {
        cid: {
            "queries": len(group),
            **{m: fsum(r[m] for r in group) / len(group) for m in METRICS},
        }
        for cid, group in grouped.items()
    }


def paired_bootstrap(differences, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    import numpy as np

    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 1000):
        stop = min(start + 1000, resamples)
        draws = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[draws].mean(1)
    low, high = np.quantile(means, [0.025, 0.975])
    verdict = "maintained" if low > 0 else "reversed" if high < 0 else "not_shown"
    return {
        "mean_difference": float(values.mean()),
        "ci95": [float(low), float(high)],
        "verdict": verdict,
        "wins_ties_losses": [
            int((values > 1e-12).sum()),
            int((abs(values) <= 1e-12).sum()),
            int((values < -1e-12).sum()),
        ],
        "queries": int(len(values)),
    }


def compare(rows, pairs=PAIRS, metrics=METRICS):
    by = {}
    for row in rows:
        by.setdefault(row["condition"], {})[row["query_id"]] = row
    result = {}
    for name, (treated, control) in pairs.items():
        if treated not in by or control not in by:
            continue
        ids = list(by[control])
        result[name] = {
            "treated": treated,
            "control": control,
            **{
                m: paired_bootstrap(
                    [by[treated][q][m] - by[control][q][m] for q in ids]
                )
                for m in metrics
            },
        }
    return result


def write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


# ------------------------------------------------------------- BC Card data
def bccard_test(data_dir):
    from evaluation.data import load_evaluation_dataset

    data_dir = Path(data_dir)
    dataset = load_evaluation_dataset(data_dir, "test", verify_all_splits=False)
    primary = read_json(data_dir / "manifest.json")["primary_slice_query_ids"]
    texts = {q.query_id: q.text for q in dataset.queries}
    return dataset, [(q, texts[q]) for q in primary], dataset.qrels


# ------------------------------------------------------------- parity checks
def parity_checkpoint(checkpoint, validation_vectors_path, data_dir, device="cuda"):
    """Check 1: adapter-loaded checkpoint reproduces PR #41 BF16 Validation vectors."""
    import numpy as np

    from evaluation.query_encoder_training import encode_queries, load_query_rows
    from retrievers.adapted_query import AdaptedQueryEmbedder
    from retrievers.dense import DenseConfig

    embedder = AdaptedQueryEmbedder(
        DenseConfig(device=device),
        base_model_path=checkpoint.base_model_path,
        checkpoint_path=checkpoint.checkpoint_path,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
    )
    model = embedder.load_model()
    texts = [
        q["text"] for q in load_query_rows(Path(data_dir) / "queries_validation.jsonl")
    ]
    vectors = encode_queries(model, texts, device=device, batch_size=16, max_tokens=512)
    reference = np.load(validation_vectors_path, allow_pickle=False)
    return {
        "check": "adapter checkpoint load vs PR #41 selected-validation-vectors (BF16, 512 tokens, batch 16)",
        "queries": len(texts),
        "bitwise_equal": bool(np.array_equal(vectors, reference)),
        "max_abs_difference": float(np.abs(vectors - reference).max()),
        "reference_sha256": file_sha256(Path(validation_vectors_path)),
    }


def parity_fixed(services, data_dir, bf16_vectors, reference_rows_path):
    """Check 2: Fixed adapter + PR #41 BF16 Test vectors reproduce PR #41 rows."""
    import numpy as np

    from fusion.minmax import UnionMinMaxFusionRetriever

    dataset, queries, qrels = bccard_test(data_dir)
    row_of = {q.query_id: i for i, q in enumerate(dataset.queries)}
    reference = {}
    for line in Path(reference_rows_path).read_text().splitlines():
        row = json.loads(line)
        reference.setdefault(row["method"], {})[row["query_id"]] = row
    service = services["bge"]
    bm25 = service.retriever(RetrievalMethod.BM25)
    dense = service.retriever(RetrievalMethod.DENSE)
    checks = {}
    for method, encoder, alpha in (
        ("baseline + Fixed alpha=0.84", "baseline", 0.84),
        ("seed43 + Fixed alpha=0.84", "seed43", 0.84),
        ("seed43 + Fixed", "seed43", 0.89),
    ):
        fusion = UnionMinMaxFusionRetriever(bm25, dense, alpha=alpha)
        vectors = np.load(bf16_vectors[encoder], allow_pickle=False)
        top_mismatch = metric_mismatch = 0
        for query_id, text in queries:
            rows, _ = fusion.rank(
                fusion.dense_scores(vectors[row_of[query_id]]),
                np.asarray(bm25.score_all(text), dtype=np.float64),
                TOP_K,
            )
            ids = [fusion._documents[r].document_id for r in rows]
            ref = reference[method][query_id]
            top_mismatch += ids != ref["top10_document_ids"]
            mine = query_metrics(ids, qrels[query_id])
            metric_mismatch += any(
                abs(mine[k] - ref[k]) > 1e-12
                for k in ("p@1", "recall@10", "mrr@10", "ndcg@10")
            )
        checks[method] = {
            "queries": len(queries),
            "top10_mismatches": top_mismatch,
            "metric_mismatches": metric_mismatch,
        }
    return {
        "check": "UnionMinMaxFusionRetriever with PR #41 BF16 Test vectors vs PR #41 per-query rows",
        "reference_sha256": file_sha256(Path(reference_rows_path)),
        "methods": checks,
        "passed": all(
            c["top10_mismatches"] == 0 and c["metric_mismatches"] == 0
            for c in checks.values()
        ),
    }


def parity_service_hybrid(services, data_dir, cache_dir, reference_rows_path):
    """Check 3: service Hybrid/Dense with cached FP32 BGE query vectors vs main replay rows."""
    import numpy as np

    from fusion.score import DEFAULT_BM25_WEIGHT, ScoreFusionRetriever
    from retrievers.dense import DenseRetriever
    from scripts.benchmark_score_fusion import _CachedEmbedder

    dataset, queries, _ = bccard_test(data_dir)
    vectors = np.load(Path(cache_dir) / "query-vectors-test.npy", allow_pickle=False)
    embedder = _CachedEmbedder(
        {
            "query_ids": [q.query_id for q in dataset.queries],
            "queries": {q.query_id: q.text for q in dataset.queries},
        },
        vectors,
    )
    service = services["bge"]
    base_dense = service.retriever(RetrievalMethod.DENSE)
    dense = DenseRetriever(
        index=base_dense.index, documents=base_dense.documents, embedder=embedder
    )
    fusion = ScoreFusionRetriever(
        service.retriever(RetrievalMethod.BM25), dense, bm25_weight=DEFAULT_BM25_WEIGHT
    )
    reference = {}
    for line in Path(reference_rows_path).read_text().splitlines():
        row = json.loads(line)
        reference.setdefault(row["method"], {})[row["query_id"]] = row
    checks = {}
    for method, retriever in (("Existing Fixed", fusion), ("Dense", dense)):
        mismatch = 0
        for query_id, text in queries:
            ids = [r.document_id for r in retriever.search(text, TOP_K)]
            mismatch += ids != reference[method][query_id]["top10_document_ids"]
        checks[method] = {"queries": len(queries), "top10_mismatches": mismatch}
    return {
        "check": "service ScoreFusionRetriever/DenseRetriever on cached FP32 BGE query vectors vs main contract replay",
        "reference_sha256": file_sha256(Path(reference_rows_path)),
        "query_vectors_sha256": file_sha256(Path(cache_dir) / "query-vectors-test.npy"),
        "methods": checks,
        "passed": all(c["top10_mismatches"] == 0 for c in checks.values()),
    }


def encoding_drift(services, queries, bf16_vectors, dataset):
    """Describe FP32 service query vectors against PR #41 BF16 vectors (no decision)."""
    import numpy as np

    row_of = {q.query_id: i for i, q in enumerate(dataset.queries)}
    out = {}
    for encoder, key in (("bge", "baseline"), ("fin", "seed43")):
        embedder = services[encoder].retriever(RetrievalMethod.DENSE).embedder
        reference = np.load(bf16_vectors[key], allow_pickle=False)
        cos = []
        for query_id, text in queries:
            v = embedder.encode_query(text)[0].astype(np.float64)
            r = reference[row_of[query_id]].astype(np.float64)
            cos.append(float(v @ r / (np.linalg.norm(v) * np.linalg.norm(r))))
        cos = np.asarray(cos)
        out[encoder] = {
            "queries": len(cos),
            "cosine_min": float(cos.min()),
            "cosine_p01": float(np.quantile(cos, 0.01)),
            "cosine_mean": float(cos.mean()),
        }
    return out


# ------------------------------------------------------------------ latency
def _proc_memory(pid="self"):
    fields = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            fields[key] = int(value.split()[0]) * 1024
    return fields


def gpu_snapshot():
    import subprocess

    def query(args):
        try:
            return (
                subprocess.run(
                    ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                .stdout.strip()
                .splitlines()
            )
        except (OSError, subprocess.CalledProcessError):
            return []

    return {
        "gpus": query(
            ["--query-gpu=index,name,utilization.gpu,memory.used,memory.total"]
        ),
        "compute_apps": query(["--query-compute-apps=pid,gpu_uuid,used_memory"]),
    }


def summarize_ms(samples):
    import numpy as np

    values = np.asarray(samples, dtype=float)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "min": float(values.min()),
        "max": float(values.max()),
        "histogram_ms": {
            "edges": [
                float(e)
                for e in np.quantile(values, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1])
            ],
        },
    }


def staged_latency(service, queries, warmup_queries, repeats=3):
    """Time service stages with the service's own functions; verify rankings match."""
    import numpy as np
    import torch

    from fusion.minmax import UnionMinMaxFusionRetriever
    from fusion.score import DEFAULT_BM25_WEIGHT, _fuse_scores
    from indexing.vectors import as_vectors

    bm25 = service.retriever(RetrievalMethod.BM25)
    dense = service.retriever(RetrievalMethod.DENSE)
    hybrid = service.retriever(RetrievalMethod.HYBRID)
    fixed = UnionMinMaxFusionRetriever(bm25, dense, alpha=SHARED_ALPHA)
    embedder = dense.embedder
    vectors32 = hybrid._vectors
    tie_ranks = hybrid._tie_ranks
    for _, text in warmup_queries:
        hybrid.search(text, TOP_K)
        fixed.search(text, TOP_K)
        dense.search(text, TOP_K)
    torch.cuda.reset_peak_memory_stats()
    stages = {
        k: []
        for k in (
            "query_encoding",
            "dense_service_hybrid",
            "dense_fixed_fp64",
            "dense_faiss",
            "bm25",
            "fusion_service_hybrid",
            "fusion_fixed",
            "total_service_hybrid",
            "total_fixed",
            "total_dense",
        )
    }
    mismatches = 0

    def timed(stage, function, *args):
        started = time.perf_counter()
        value = function(*args)
        stages[stage].append((time.perf_counter() - started) * 1e3)
        return value

    def service_fusion(scores, sparse):
        fused = _fuse_scores(
            scores,
            sparse.astype(np.float64),
            maximum=float(np.max(sparse)),
            bm25_weight=DEFAULT_BM25_WEIGHT,
        )
        return np.lexsort((tie_ranks, -fused))[:TOP_K]

    for _ in range(repeats):
        for _, text in queries:
            vector = timed(
                "query_encoding", lambda: as_vectors(embedder.encode_query(text))
            )
            scores = timed(
                "dense_service_hybrid",
                lambda: (vectors32 @ vector[0]).astype(np.float64),
            )
            dense64 = timed("dense_fixed_fp64", fixed.dense_scores, vector[0])
            faiss_hits = timed("dense_faiss", dense.index.search, vector, TOP_K)
            sparse = timed("bm25", bm25.score_all, text)
            selected = timed("fusion_service_hybrid", service_fusion, scores, sparse)
            fixed_rows, _ = timed(
                "fusion_fixed",
                fixed.rank,
                dense64,
                np.asarray(sparse, dtype=np.float64),
                TOP_K,
            )
            hybrid_hits = timed("total_service_hybrid", hybrid.search, text, TOP_K)
            fixed_hits = timed("total_fixed", fixed.search, text, TOP_K)
            dense_hits = timed("total_dense", dense.search, text, TOP_K)
            documents = hybrid._documents
            mismatches += (
                [documents[r].document_id for r in selected]
                != [h.document_id for h in hybrid_hits]
                or [documents[r].document_id for r in fixed_rows]
                != [h.document_id for h in fixed_hits]
                or [documents[r].document_id for r, _ in faiss_hits]
                != [h.document_id for h in dense_hits]
            )
    return {
        "repeats": repeats,
        "queries": len(queries),
        "warmup_queries": len(warmup_queries),
        "staged_ranking_mismatches": mismatches,
        "stages_ms": {k: summarize_ms(v) for k, v in stages.items()},
        "torch_cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "process_memory": _proc_memory(),
    }


def http_latency(
    base_url, queries, warmup_queries, method, repeats=3, concurrency=(1, 4)
):
    """Measure full POST /search latency and throughput against a running server."""
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    with httpx.Client(base_url=base_url, timeout=120) as client:
        for _, text in warmup_queries:
            client.post(
                "/search", json={"query": text, "method": method, "top_k": TOP_K}
            ).raise_for_status()
        samples = []
        for _ in range(repeats):
            for _, text in queries:
                t = time.perf_counter()
                client.post(
                    "/search", json={"query": text, "method": method, "top_k": TOP_K}
                ).raise_for_status()
                samples.append((time.perf_counter() - t) * 1e3)
    throughput = {}
    for workers in concurrency:

        def one(text):
            with httpx.Client(base_url=base_url, timeout=120) as c:
                t = time.perf_counter()
                c.post(
                    "/search", json={"query": text, "method": method, "top_k": TOP_K}
                ).raise_for_status()
                return (time.perf_counter() - t) * 1e3

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            lat = list(pool.map(one, [text for _, text in queries]))
        elapsed = time.perf_counter() - started
        throughput[str(workers)] = {
            "requests": len(lat),
            "seconds": elapsed,
            "requests_per_second": len(lat) / elapsed,
            "latency_ms": summarize_ms(lat),
        }
    return {
        "method": method,
        "sequential_ms": summarize_ms(samples),
        "throughput": throughput,
    }


# ----------------------------------------------------------------- external
def load_external(cache_dir, name):
    import numpy as np

    directory = Path(cache_dir) / name
    data = read_json(directory / "data.json")
    documents = [
        Document(d["document_id"], d["text"], d.get("title")) for d in data["documents"]
    ]
    vectors = np.load(directory / "document-vectors.npy", allow_pickle=False)
    query_vectors = np.load(directory / "query-vectors.npy", allow_pickle=False)
    split_ids = data["splits"]["dev"] if name == "miracl10k" else data["query_ids"]
    queries = [(q, data["queries"][q]) for q in split_ids]
    qrels = {q: data["qrels"][q] for q in split_ids}
    return data, documents, vectors, query_vectors, queries, qrels


def environment():
    import platform

    import torch

    from evaluation.benchmark import _code_provenance

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "code": _code_provenance(),
    }


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

"""Reproducible BC Card QDAP inputs and the eight-method evaluation contract."""

import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation.json_io import read_json, write_json
from evaluation.qdap import ALPHAS, PoolBatch
from evaluation.qdap_training import audit_official_partitions

METHODS = (
    "BM25",
    "Dense",
    "Existing Fixed",
    "Existing Router",
    "Paper Fixed",
    "QDAP-S",
    "QDAP-L",
    "Oracle alpha",
)


def rows(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def primary(query):
    return query["provenance"] in ("base", "faq_direct")


def legacy_ranking(dense, sparse, ids, weight):
    """Keep main's full-corpus cosine + w * BM25/corpus-max and ID ties."""
    dense, sparse = np.asarray(dense, float), np.asarray(sparse, float)
    maximum = float(sparse.max())
    score = dense + weight * sparse / maximum if maximum > 0 else dense
    return np.lexsort((ids, -score))[:10]


def batch_utilities(pools, judgments):
    """Vectorize all 101 graded nDCG targets; IDCG includes outside documents."""
    grade = np.array(
        [
            [q.get(int(j), 0) if j >= 0 else 0 for j in p]
            for p, q in zip(pools.indices, judgments, strict=True)
        ]
    )
    ideal = np.array(
        [
            sum(
                (2**v - 1) / np.log2(j + 2)
                for j, v in enumerate(sorted(q.values(), reverse=True)[:10])
            )
            for q in judgments
        ]
    )
    scores = ALPHAS[None, :, None] * pools.pairs[:, None, :, 0]
    scores += (1 - ALPHAS[None, :, None]) * pools.pairs[:, None, :, 1]
    scores = np.where(pools.valid[:, None], scores, -np.inf)
    order = np.argsort(-scores, axis=-1, kind="stable")[:, :, :10]
    gains = np.take_along_axis(2 ** grade[:, None] - 1, order, axis=-1)
    # The top-k width can be smaller than ten in test fixtures.
    dcg = (gains / np.log2(np.arange(order.shape[-1]) + 2)).sum(-1)
    return np.divide(
        dcg, ideal[:, None], out=np.zeros_like(dcg), where=ideal[:, None] > 0
    )


def judgments_for(data, split, queries, documents):
    by_id = {row["document_id"]: i for i, row in enumerate(documents)}
    qrels = {q["query_id"]: {} for q in queries}
    for row in rows(data / f"qrels_{split}.jsonl"):
        if row["query_id"] in qrels:
            qrels[row["query_id"]][by_id[row["document_id"]]] = row["relevance"]
    if any(not q for q in qrels.values()):
        raise ValueError("every query needs qrels")
    return [qrels[q["query_id"]] for q in queries]


def load_pools(out, split):
    with np.load(out / f"{split}.npz", allow_pickle=False) as z:
        return PoolBatch(z["pools"], z["pairs"], z["valid"])


def prepare(inputs, out):
    """Prepare only Train/Validation labels, preserving every official Train row."""
    from data.loader import Document
    from retrievers.bm25 import BM25Retriever

    out.mkdir(parents=True, exist_ok=False)
    data = inputs / "data/processed/bccard"
    cache = inputs / "artifacts/bccard-adaptive-vectors-resumed"
    manifest = read_json(data / "manifest.json")
    encoding = read_json(cache / "encoding.json")
    for file, digest in manifest["output_sha256"].items():
        if sha(data / file) != digest:
            raise ValueError(f"dataset hash mismatch: {file}")
    if sha(data / "manifest.json") != encoding["dataset_manifest_sha256"]:
        raise ValueError("dataset/vector manifest mismatch")
    for file, digest in encoding["vector_sha256"].items():
        if sha(cache / file) != digest:
            raise ValueError(f"vector hash mismatch: {file}")
    docs = rows(data / "corpus.jsonl")
    ids = np.array([d["document_id"] for d in docs])
    queries = {
        s: rows(data / f"queries_{s}.jsonl") for s in ("train", "validation", "test")
    }
    audit = audit_official_partitions(**queries)
    vector = np.load(cache / "document-vectors.npy").astype(float)
    bm25 = BM25Retriever([Document(**doc) for doc in docs])
    saved_scores = inputs / "artifacts/paper-guided/dat-local/bccard"
    archived_queries = read_json(saved_scores / "queries.json")
    lookup = {q["query_id"]: i for i, q in enumerate(archived_queries)}
    dense_cache = np.load(saved_scores / "dense.npy", mmap_mode="r")
    sparse_cache = np.load(saved_scores / "sparse.npy", mmap_mode="r")
    for split in ("train", "validation"):
        source = queries[split]
        if [q["query_id"] for q in source] != encoding["splits"][split]["query_ids"]:
            raise ValueError("query vector order mismatch")
        selected = (
            np.arange(len(source))
            if split == "train"
            else np.flatnonzero([primary(q) for q in source])
        )
        qs = [source[i] for i in selected]
        labels = judgments_for(data, split, qs, docs)
        qvec = np.load(cache / f"query-vectors-{split}.npy")[selected]
        count = len(qs)
        pool = np.full((count, 200), -1, dtype=np.int64)
        pairs = np.zeros((count, 200, 2), dtype=np.float64)
        valid = np.zeros((count, 200), bool)
        quality = np.zeros((count, 101), dtype=np.float64)
        for start in range(0, count, 64):
            end = min(start + 64, count)
            if split == "train":
                dense = qvec[start:end].astype(float) @ vector.T
                sparse = np.stack(
                    [bm25.score_all(q["text"]).astype(float) for q in qs[start:end]]
                )
            else:
                ix = [lookup[q["query_id"]] for q in qs[start:end]]
                dense, sparse = (
                    np.asarray(dense_cache[ix]),
                    np.asarray(sparse_cache[ix]),
                )
            batch = PoolBatch.from_scores(dense, sparse, ids)
            pool[start:end], pairs[start:end], valid[start:end] = (
                batch.indices,
                batch.pairs,
                batch.valid,
            )
            quality[start:end] = batch_utilities(batch, labels[start:end])
            if start % 2048 == 0:
                print({"prepare": split, "rows": start, "total": count}, flush=True)
        np.savez(
            out / f"{split}.npz",
            pools=pool,
            pairs=pairs,
            valid=valid,
            quality=quality,
            vectors=qvec,
        )
        write_json(out / f"{split}-queries.json", qs)
    write_json(
        out / "protocol.json",
        {
            "inputs": str(inputs),
            "audit": audit,
            "train": "all official Train rows, no calibration or resplit",
            "selection": "official Validation non-synthetic nDCG; epoch, CE, seed, mean/argmax; stable ties",
            "test": "721 non-synthetic official Test; previously exposed; not a fresh test",
            "architecture": "Linear1024x101, Conv1d(1,1,7,padding3), softmax; L independent original BGE-M3 CLS/L2",
            "optimizer": "AdamW lr1e-4,wd.01,5epochs,effective batch128,OneCycle warmup.05,clip1",
            "large_resource_adaptation": "BF16 encoder/Adam states, FP32 head, gradient checkpointing, microbatch8,max128 tokens,seed42",
            "small": "frozen original BGE-M3 vectors; seeds42/43/44; both CE modes",
            "loss": ".62 squared-target CE + .38 summed CDF WD; literal unnormalized or reference renormalized CE",
            "candidate_pool": "Dense/BM25 top100 union, per-channel union minmax; ID ties; 101 alpha bins",
            "reference_commit": "9e644b6362e5d429b7b4154370efec2d36b98691",
            "input_sha256": {
                str(p): sha(p)
                for p in [
                    data / "manifest.json",
                    cache / "encoding.json",
                    cache / "document-vectors.npy",
                    cache / "query-vectors-train.npy",
                    cache / "query-vectors-validation.npy",
                ]
            },
            "prepared_sha256": {
                name: sha(out / name)
                for name in [
                    "train.npz",
                    "validation.npz",
                    "train-queries.json",
                    "validation-queries.json",
                ]
            },
        },
    )
    print({"prepared": str(out), "audit": audit}, flush=True)

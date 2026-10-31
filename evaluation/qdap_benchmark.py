"""Reproducible BC Card QDAP inputs and the eight-method evaluation contract."""

import numpy as np

from evaluation.json_io import read_json, write_json
from evaluation.qdap import ALPHAS, PoolBatch
from evaluation.qdap_artifacts import METHODS as METHODS
from evaluation.qdap_artifacts import hashes, provenance
from evaluation.qdap_artifacts import primary as primary
from evaluation.qdap_artifacts import rows as rows
from evaluation.qdap_artifacts import sha as sha
from evaluation.qdap_training import audit_official_partitions


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


def score_batches(queries, query_vectors, document_vectors, bm25, *, batch_size=64):
    """Use one float64 Dense/BM25 scoring rule for every official split."""
    if len(queries) != len(query_vectors) or not queries or batch_size <= 0:
        raise ValueError("aligned nonempty queries and positive batch size required")
    vectors = np.asarray(document_vectors, dtype=np.float64)
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        dense = np.asarray(query_vectors[start:end], dtype=np.float64) @ vectors.T
        sparse = np.stack(
            [bm25.score_all(q["text"]) for q in queries[start:end]]
        ).astype(np.float64)
        yield start, end, dense, sparse


def split_scores(inputs, split, queries, vector_cache=None):
    """Generate scores from shared frozen vectors and BM25, preserving query IDs."""
    from data.loader import Document
    from evaluation.enterprise import _ids_digest
    from retrievers.bm25 import BM25Retriever

    data = inputs / "data/processed/bccard"
    cache = vector_cache or inputs / "artifacts/bccard-adaptive-vectors-resumed"
    record = read_json(cache / "encoding.json")
    if sha(data / "manifest.json") != record["dataset_manifest_sha256"]:
        raise ValueError("dataset/vector manifest mismatch")
    official = rows(data / f"queries_{split}.jsonl")
    if [q["query_id"] for q in official] != record["splits"][split]["query_ids"]:
        raise ValueError("query vector order mismatch")
    lookup = {q["query_id"]: i for i, q in enumerate(official)}
    if len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("duplicate requested query IDs")
    for q in queries:
        if (
            q["query_id"] not in lookup
            or q["text"] != official[lookup[q["query_id"]]]["text"]
        ):
            raise ValueError("requested query differs from official split")
    docs = rows(data / "corpus.jsonl")
    if (
        record.get("document_ids_sha256") is not None
        and _ids_digest(d["document_id"] for d in docs) != record["document_ids_sha256"]
    ):
        raise ValueError("document vector order mismatch")
    for name in ("document-vectors.npy", f"query-vectors-{split}.npy"):
        if sha(cache / name) != record["vector_sha256"][name]:
            raise ValueError(f"vector hash mismatch: {name}")
    vectors = np.load(cache / "document-vectors.npy", allow_pickle=False)
    qvectors = np.load(cache / f"query-vectors-{split}.npy", allow_pickle=False)
    if len(vectors) != len(docs) or len(qvectors) != len(official):
        raise ValueError("vector rows do not match corpus/query order")
    qvectors = qvectors[[lookup[q["query_id"]] for q in queries]]
    bm25 = BM25Retriever([Document(**doc) for doc in docs])
    batches = list(score_batches(queries, qvectors, vectors, bm25))
    return np.concatenate([b[2] for b in batches]), np.concatenate(
        [b[3] for b in batches]
    )


def prepare(inputs, out, vector_cache=None):
    """Prepare only Train/Validation labels, preserving every official Train row."""
    from data.loader import Document
    from retrievers.bm25 import BM25Retriever

    out.mkdir(parents=True, exist_ok=False)
    data = inputs / "data/processed/bccard"
    cache = vector_cache or inputs / "artifacts/bccard-adaptive-vectors-resumed"
    manifest = read_json(data / "manifest.json")
    encoding = read_json(cache / "encoding.json")
    for file in (
        "corpus.jsonl",
        "queries_train.jsonl",
        "queries_validation.jsonl",
        "queries_test.jsonl",
        "qrels_train.jsonl",
        "qrels_validation.jsonl",
    ):
        digest = manifest["output_sha256"][file]
        if sha(data / file) != digest:
            raise ValueError(f"dataset hash mismatch: {file}")
    if sha(data / "manifest.json") != encoding["dataset_manifest_sha256"]:
        raise ValueError("dataset/vector manifest mismatch")
    for file in (
        "document-vectors.npy",
        "query-vectors-train.npy",
        "query-vectors-validation.npy",
    ):
        digest = encoding["vector_sha256"][file]
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
        pool = np.full((count, min(200, len(ids))), -1, dtype=np.int64)
        pairs = np.zeros((count, min(200, len(ids)), 2), dtype=np.float64)
        valid = np.zeros((count, min(200, len(ids))), bool)
        quality = np.zeros((count, 101), dtype=np.float64)
        for start, end, dense, sparse in score_batches(qs, qvec, vector, bm25):
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
        if split == "validation":
            write_json(out / "validation-qrels.json", labels)
    write_json(
        out / "protocol.json",
        {
            "schema_version": 2,
            "run_id": out.name,
            "encoding": encoding,
            "environment": provenance(),
            "audit": audit,
            "train": "all official Train rows, no calibration or resplit",
            "qrels_provenance": {
                split: manifest.get("split_stats", {})
                .get(split, {})
                .get("qrels_source", "not recorded in dataset manifest")
                for split in ("train", "validation", "test")
            },
            "qrels_rule_verification": "BC Card preparation checks that the dataset-card chunk-distance rule reproduces the official Validation/Test labels",
            "training_target": "softmax over alpha-specific graded nDCG using Train qrels",
            "selection": "official Validation non-synthetic nDCG; epoch, CE, seed, mean/argmax; stable ties",
            "test": "721 non-synthetic official Test; previously exposed; not a fresh test",
            "architecture": "Linear1024x101, Conv1d(1,1,7,padding3), softmax; L independent original BGE-M3 CLS/L2",
            "optimizer": "AdamW lr1e-4,wd.01,5epochs,effective batch128,OneCycle warmup.05,clip1",
            "large_resource_adaptation": "BF16 encoder/Adam states, FP32 head, gradient checkpointing, microbatch8,max128 tokens,seed42",
            "small": "frozen original BGE-M3 vectors; seeds42/43/44; both CE modes",
            "loss": ".62 squared-target CE + .38 summed CDF WD; literal unnormalized or reference renormalized CE",
            "candidate_pool": "Dense/BM25 top100 union, per-channel union minmax; ID ties; 101 alpha bins",
            "reference_commit": "9e644b6362e5d429b7b4154370efec2d36b98691",
            "input_sha256": hashes(
                {
                    **{
                        f"data/{name}": data / name
                        for name in (
                            "manifest.json",
                            "corpus.jsonl",
                            "queries_train.jsonl",
                            "queries_validation.jsonl",
                            "queries_test.jsonl",
                            "qrels_train.jsonl",
                            "qrels_validation.jsonl",
                        )
                    },
                    **{
                        f"vectors/{name}": cache / name
                        for name in (
                            "encoding.json",
                            "document-vectors.npy",
                            "query-vectors-train.npy",
                            "query-vectors-validation.npy",
                        )
                    },
                }
            ),
            "prepared_sha256": {
                name: sha(out / name)
                for name in [
                    "train.npz",
                    "validation.npz",
                    "train-queries.json",
                    "validation-queries.json",
                    "validation-qrels.json",
                ]
            },
        },
    )
    write_json(
        out / "config.json",
        {
            "schema_version": 2,
            "run_id": out.name,
            "variants": ["S", "L"],
            "ce_modes": ["literal", "reference"],
            "seeds": {"S": [42, 43, 44], "L": [42]},
            "epochs": 5,
            "alpha_bins": 101,
            "readouts": ["mean", "argmax"],
        },
    )
    write_json(
        out / "manifest.json",
        {
            "schema_version": 2,
            "run_id": out.name,
            "files": hashes(
                {name: out / name for name in ("config.json", "protocol.json")}
            ),
            "stages": {
                "train": [
                    "protocol.json",
                    "train.npz",
                    "validation.npz",
                    "train-queries.json",
                    "validation-queries.json",
                    "validation-qrels.json",
                ],
                "evaluate": [
                    "selection.json",
                    "data/test",
                    "vectors/test",
                    "router",
                    "model",
                    "selected_checkpoints",
                ],
                "report": [
                    "per-query.jsonl",
                    "per-query-manifest.json",
                    "selection.json",
                    "protocol.json",
                    "data/test",
                ],
            },
        },
    )
    print({"prepared": str(out), "audit": audit}, flush=True)

"""Evaluate Phase 3 baselines on the frozen BC Card protocol without retuning them.

Vectors are encoded once with the pinned BGE-M3 revision and reused, so latency
follows the Phase 3 final scope: cached-vector CPU retrieval and fusion. Query
encoding time is recorded separately during encoding.
"""

import re
from collections import Counter
from datetime import UTC, datetime
from math import fsum
from pathlib import Path
from time import perf_counter

from data.bccard import NON_SYNTHETIC, load_query_metadata
from data.preprocess import build_index_text
from evaluation.benchmark import _code_provenance, _environment, benchmark_retriever
from evaluation.constants import DEFAULT_REPEATS, DEFAULT_WARMUP, EVALUATION_TOP_K
from evaluation.data import file_sha256, load_evaluation_dataset
from evaluation.json_io import read_json, write_json
from evaluation.latency import latency_summary
from evaluation.metrics import graded_ndcg_at_k, precision_at_k
from retrievers.base import SearchResult

BASELINE_METHODS = ("bm25", "dense", "score_fusion")
ENCODED_SPLITS = ("validation", "test")
REPORT_METRICS = ("p@1", "recall@10", "mrr@10", "ndcg@10", "graded_ndcg@10")
_TIE_TOLERANCE = 1e-12


# Phase 5 외부 encoder는 model card 설정 그대로 쓴다(prefix 없음, full dimension).
ENCODERS = {
    "bge": {"model_name": "BAAI/bge-m3"},
    "kanana": {
        "model_name": "kakaocorp/kanana-2-3b-embedding",
        "revision": "2c85dbe61af8ff033d2c9d46a686c665153069aa",
        "trust_remote_code": True,
    },
}


def encode_bccard(
    data_dir, cache_dir, *, splits=ENCODED_SPLITS, device="cuda:0", encoder="bge"
):
    """Encode the corpus and selected query splits one text at a time."""
    import numpy as np

    from retrievers.dense import DenseConfig, DenseEmbedder

    if encoder not in ENCODERS:
        raise ValueError(f"encoder는 {', '.join(ENCODERS)} 중에서 선택합니다.")
    spec = ENCODERS[encoder]
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise ValueError("cache-dir은 새 경로 또는 비어 있는 디렉터리여야 합니다.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    datasets = {split: load_evaluation_dataset(data_dir, split) for split in splits}
    documents = next(iter(datasets.values())).documents
    config = DenseConfig(
        model_name=spec["model_name"],
        revision=spec.get("revision"),
        device=device,
        batch_size=1,
    )
    embedder = DenseEmbedder(config)
    trust_remote_code = spec.get("trust_remote_code", False)
    if trust_remote_code and embedder.model is None:
        from sentence_transformers import SentenceTransformer

        embedder.model = SentenceTransformer(
            config.model_name,
            device=device,
            revision=config.revision,
            trust_remote_code=True,
        )
    on_gpu = str(device).startswith("cuda")
    if on_gpu:
        import torch

        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    vectors = np.concatenate(
        [embedder.encode_documents([build_index_text(doc)]) for doc in documents]
    )
    document_seconds = perf_counter() - started
    np.save(cache_dir / "document-vectors.npy", vectors, allow_pickle=False)
    record = {
        "model": {
            "model_name": config.model_name,
            "revision": config.revision,
            "normalize_embeddings": config.normalize_embeddings,
            "query_prefix": config.query_prefix,
            "passage_prefix": config.passage_prefix,
            "batch_size": config.batch_size,
            "device": device,
            "trust_remote_code": trust_remote_code,
        },
        "max_seq_length": embedder.model.max_seq_length,
        "dtype": str(next(embedder.model.parameters()).dtype),
        "embedding_dimension": int(vectors.shape[1]),
        "libraries": _library_versions(),
        "preprocessing": {"document": "Pair positive text only", "query": "unchanged"},
        "dataset_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "document_ids_sha256": _ids_digest(doc.document_id for doc in documents),
        "document_encoding_seconds": document_seconds,
        "splits": {},
        "vector_sha256": {"document-vectors.npy": None},
    }
    for split, dataset in datasets.items():
        rows, timings = [], []
        for query in dataset.queries:
            started = perf_counter()
            rows.append(embedder.encode_query(query.text))
            timings.append((perf_counter() - started) * 1000)
        name = f"query-vectors-{split}.npy"
        np.save(cache_dir / name, np.concatenate(rows), allow_pickle=False)
        record["splits"][split] = {
            "query_ids": [query.query_id for query in dataset.queries],
            "query_encoding_latency": latency_summary(timings),
        }
        record["vector_sha256"][name] = None
    for name in record["vector_sha256"]:
        record["vector_sha256"][name] = file_sha256(cache_dir / name)
    if on_gpu:
        record["gpu"] = {
            "name": torch.cuda.get_device_name(device),
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }
    write_json(cache_dir / "encoding.json", record)
    return record


def _library_versions():
    import importlib.metadata

    versions = {}
    for name in ("torch", "transformers", "sentence-transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _ids_digest(ids):
    import hashlib

    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _load_vectors(data_dir, cache_dir, dataset):
    """Load cached vectors only when they match this exact prepared dataset."""
    import numpy as np

    from indexing.vectors import as_vectors

    record = read_json(Path(cache_dir) / "encoding.json")
    if record["dataset_manifest_sha256"] != file_sha256(
        Path(data_dir) / "manifest.json"
    ):
        raise ValueError("vector cache가 다른 dataset manifest로 생성되었습니다.")
    for name, digest in record["vector_sha256"].items():
        if file_sha256(Path(cache_dir) / name) != digest:
            raise ValueError(f"vector cache SHA-256 불일치: {name}")
    if record["document_ids_sha256"] != _ids_digest(
        doc.document_id for doc in dataset.documents
    ):
        raise ValueError("vector cache의 문서 순서가 corpus와 다릅니다.")
    split = record["splits"].get(dataset.split)
    if split is None or split["query_ids"] != [q.query_id for q in dataset.queries]:
        raise ValueError(f"{dataset.split} 질의 vector가 cache에 없습니다.")
    documents = as_vectors(
        np.load(Path(cache_dir) / "document-vectors.npy", allow_pickle=False)
    )
    queries = as_vectors(
        np.load(
            Path(cache_dir) / f"query-vectors-{dataset.split}.npy", allow_pickle=False
        ),
        documents.shape[1],
    )
    return documents, queries, record


def _build_retrievers(dataset, vectors, query_vectors):
    from fusion.score import DEFAULT_BM25_WEIGHT, ScoreFusionRetriever
    from indexing.faiss_index import FaissIndex
    from retrievers.bm25 import DEFAULT_B, DEFAULT_K1, SCORING_METHOD, BM25Retriever
    from retrievers.dense import DenseRetriever
    from scripts.benchmark_score_fusion import _CachedEmbedder

    started = perf_counter()
    sparse = BM25Retriever(dataset.documents)
    sparse_seconds = perf_counter() - started
    lookup = {
        "query_ids": [query.query_id for query in dataset.queries],
        "queries": {query.query_id: query.text for query in dataset.queries},
    }
    started = perf_counter()
    dense = DenseRetriever(
        index=FaissIndex.build(vectors),
        documents=dataset.documents,
        embedder=_CachedEmbedder(lookup, query_vectors),
    )
    dense_seconds = perf_counter() - started
    started = perf_counter()
    fusion = ScoreFusionRetriever(sparse, dense, bm25_weight=DEFAULT_BM25_WEIGHT)
    fusion_seconds = sparse_seconds + dense_seconds + perf_counter() - started
    return {
        "bm25": (
            sparse,
            sparse_seconds,
            {
                "analyzer": "char_wb",
                "ngram_range": [2, 4],
                "k1": DEFAULT_K1,
                "b": DEFAULT_B,
                "method": SCORING_METHOD,
            },
        ),
        "dense": (dense, dense_seconds, {"embedding": "cached BGE-M3 vectors"}),
        "score_fusion": (
            fusion,
            fusion_seconds,
            {
                "bm25_weight": DEFAULT_BM25_WEIGHT,
                "normalization": "per-query corpus maximum",
                "candidate_scope": "full corpus",
                "vector_correction": False,
            },
        ),
    }


def query_metrics(results, judgments):
    """Add P@1 and graded nDCG@10 to the shared per-query metrics."""
    hits = [SearchResult(**hit) for hit in results]
    return {
        "p@1": precision_at_k(hits, judgments, 1),
        "graded_ndcg@10": graded_ndcg_at_k(hits, judgments, EVALUATION_TOP_K),
    }


def slice_definitions(metadata):
    """Return descriptive slices from explicit dataset metadata only."""
    slices = {
        "all": list(metadata),
        "non_synthetic": [
            q for q, row in metadata.items() if row["provenance"] in NON_SYNTHETIC
        ],
        "synthetic": [
            q for q, row in metadata.items() if row["provenance"] not in NON_SYNTHETIC
        ],
    }
    if all("text_in_train" in row for row in metadata.values()):
        slices["text_not_in_train"] = [
            q for q, row in metadata.items() if not row["text_in_train"]
        ]
    for field in ("provenance", "page_type", "site"):
        values = sorted({row[field] for row in metadata.values() if row[field]})
        for value in values:
            slices[f"{field}={value}"] = [
                q for q, row in metadata.items() if row[field] == value
            ]
    return slices


def summarize(per_query, query_ids):
    return {
        "query_count": len(query_ids),
        **{
            name: fsum(per_query[q][name] for q in query_ids) / len(query_ids)
            for name in REPORT_METRICS
        },
    }


def run_enterprise_baselines(
    data_dir,
    cache_dir,
    split,
    *,
    methods=BASELINE_METHODS,
    warmup=DEFAULT_WARMUP,
    repeats=DEFAULT_REPEATS,
    progress=None,
):
    """Evaluate fixed baselines on one split and summarize every metadata slice."""
    if not methods or any(name not in BASELINE_METHODS for name in methods):
        raise ValueError("methods는 bm25, dense, score_fusion 중에서 선택합니다.")
    if split == "train":
        raise ValueError("train은 router 학습용이며 baseline 보고 대상이 아닙니다.")
    dataset = load_evaluation_dataset(data_dir, split)
    vectors, query_vectors, encoding = _load_vectors(data_dir, cache_dir, dataset)
    retrievers = _build_retrievers(dataset, vectors, query_vectors)
    metadata = load_query_metadata(data_dir, split)
    slices = slice_definitions(metadata)
    report = {
        "schema_version": "1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {"split": split, **dataset.provenance},
        "conditions": {
            "top_k": EVALUATION_TOP_K,
            "warmup_passes": warmup,
            "measurement_passes": repeats,
            "latency_scope": "cached-vector CPU retrieval and fusion; excludes query encoding",
            "relevance": "binary metrics use relevance > 0; graded nDCG uses gains 1-3",
            "quality_run": "first measured pass",
            "retuned": False,
        },
        "encoding": {
            key: encoding[key]
            for key in (
                "model",
                "max_seq_length",
                "dtype",
                "embedding_dimension",
                "vector_sha256",
            )
            if key in encoding
        }
        | {
            "record_sha256": file_sha256(Path(cache_dir) / "encoding.json"),
            "query_encoding_latency": encoding["splits"][split][
                "query_encoding_latency"
            ],
        },
        "environment": _environment(),
        "code": _code_provenance(),
        "slices": {name: len(ids) for name, ids in slices.items() if ids},
        "methods": {},
    }
    report["code"]["sha256"]["evaluation/enterprise.py"] = file_sha256(Path(__file__))
    for name in methods:
        retriever, setup_seconds, config = retrievers[name]
        if name == "dense":
            config = {"embedding": f"cached {encoding['model']['model_name']} vectors"}
        if progress:
            progress(f"{split}: {name} / {len(dataset.queries)} queries / {repeats}")
        result = benchmark_retriever(retriever, dataset, warmup=warmup, repeats=repeats)
        per_query = {}
        for row in result["queries"]:
            row["metrics"] |= query_metrics(
                row["results"], dataset.qrels[row["query_id"]]
            )
            row["metrics"]["mrr@10"] = row["metrics"]["rr@10"]
            per_query[row["query_id"]] = row["metrics"]
        report["methods"][name] = {
            "config": config,
            "setup_seconds": setup_seconds,
            "latency": result["latency"],
            "slices": {
                slice_name: summarize(per_query, ids)
                for slice_name, ids in slices.items()
                if ids
            },
            "queries": result["queries"],
        }
    return report


def preference_analysis(report, slices, metric="ndcg@10"):
    """Describe per-query method preferences on a non-test split.

    The oracle picks the best existing method per query; it is a headroom
    estimate for routing, not a method, and is refused on the held-out test.
    """
    if report["dataset"]["split"] == "test":
        raise ValueError(
            "oracle/preference 분석은 held-out test에서 계산하지 않습니다."
        )
    if set(report["methods"]) != set(BASELINE_METHODS):
        raise ValueError("세 baseline 결과가 모두 필요합니다.")
    scores = {
        method: {row["query_id"]: row["metrics"][metric] for row in data["queries"]}
        for method, data in report["methods"].items()
    }
    bm25, dense, fusion = (scores[name] for name in BASELINE_METHODS)

    def share(predicate, ids):
        return sum(predicate(q) for q in ids) / len(ids)

    def close(a, b):
        return abs(a - b) <= _TIE_TOLERANCE

    analysis = {}
    for slice_name, ids in slices.items():
        if not ids:
            continue
        means = {
            name: fsum(scores[name][q] for q in ids) / len(ids)
            for name in BASELINE_METHODS
        }
        best_fixed = max(means, key=means.get)
        oracle = fsum(max(scores[name][q] for name in BASELINE_METHODS) for q in ids)
        oracle /= len(ids)
        analysis[slice_name] = {
            "query_count": len(ids),
            "bm25_gt_dense": share(lambda q: bm25[q] > dense[q] + _TIE_TOLERANCE, ids),
            "dense_gt_bm25": share(lambda q: dense[q] > bm25[q] + _TIE_TOLERANCE, ids),
            "bm25_dense_tied": share(lambda q: close(bm25[q], dense[q]), ids),
            "fusion_gt_both": share(
                lambda q: fusion[q] > max(bm25[q], dense[q]) + _TIE_TOLERANCE, ids
            ),
            "fusion_lt_best_component": share(
                lambda q: fusion[q] < max(bm25[q], dense[q]) - _TIE_TOLERANCE, ids
            ),
            "all_three_tied": share(
                lambda q: close(bm25[q], dense[q]) and close(dense[q], fusion[q]), ids
            ),
            "means": means,
            "best_fixed": best_fixed,
            "oracle_bm25_dense": fsum(max(bm25[q], dense[q]) for q in ids) / len(ids),
            "oracle_three": oracle,
            "oracle_gain_over_best_fixed": oracle - means[best_fixed],
        }
    return {"metric": metric, "split": report["dataset"]["split"], "slices": analysis}


def write_summary(report, path):
    """Write metrics without per-query rankings for documentation."""
    summary = {key: value for key, value in report.items() if key != "methods"}
    summary["methods"] = {
        name: {key: value for key, value in data.items() if key != "queries"}
        for name, data in report["methods"].items()
    }
    write_json(Path(path), summary)


MATERIAL_MARGIN = 0.2
_ABBREVIATION = re.compile(r"[A-Z]{2,}")
_LATIN = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"\d")


def _bigrams(text):
    compact = "".join(text.split())
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def query_properties(text, positive_text):
    """Observable query properties only; no hand-made semantic taxonomy."""
    grams = _bigrams(text)
    return {
        "has_digit": bool(_DIGIT.search(text)),
        "has_latin": bool(_LATIN.search(text)),
        "has_uppercase_abbreviation": bool(_ABBREVIATION.search(text)),
        "length_chars": len(text),
        "bigram_overlap_with_positive": (
            len(grams & _bigrams(positive_text)) / len(grams) if grams else 0.0
        ),
    }


def validation_handoff(report, metadata, qrels, documents, metric="ndcg@10"):
    """Win counts, oracle metrics, lexical profiles and examples for Issue #24.

    Validation only. The oracle chooses per query the method with the best
    `metric` (ties keep score_fusion, then dense) and reports
    that method's other metrics; it is an upper-bound estimate, not a method.
    """
    if report["dataset"]["split"] == "test":
        raise ValueError("handoff 분석은 held-out test에서 계산하지 않습니다.")
    rows = {
        method: {row["query_id"]: row for row in data["queries"]}
        for method, data in report["methods"].items()
    }
    query_ids = list(rows["bm25"])
    value = {m: {q: rows[m][q]["metrics"][metric] for q in query_ids} for m in rows}
    bm25, dense, fusion = (value[name] for name in BASELINE_METHODS)
    count = len(query_ids)

    def tally(predicate):
        hits = sum(predicate(q) for q in query_ids)
        return {"count": hits, "fraction": hits / count}

    gt = lambda a, b: a > b + _TIE_TOLERANCE  # noqa: E731
    wins = {
        "bm25_gt_dense": tally(lambda q: gt(bm25[q], dense[q])),
        "dense_gt_bm25": tally(lambda q: gt(dense[q], bm25[q])),
        "bm25_eq_dense": tally(
            lambda q: not gt(bm25[q], dense[q]) and not gt(dense[q], bm25[q])
        ),
        "fusion_gt_dense": tally(lambda q: gt(fusion[q], dense[q])),
        "fusion_lt_dense": tally(lambda q: gt(dense[q], fusion[q])),
        "fusion_eq_dense": tally(
            lambda q: not gt(fusion[q], dense[q]) and not gt(dense[q], fusion[q])
        ),
        "all_three_tied": tally(
            lambda q: (
                max(bm25[q], dense[q], fusion[q]) - min(bm25[q], dense[q], fusion[q])
                <= _TIE_TOLERANCE
            )
        ),
    }
    for name in BASELINE_METHODS:
        others = [m for m in BASELINE_METHODS if m != name]
        wins[f"{name}_uniquely_best"] = tally(
            lambda q, n=name, o=others: all(gt(value[n][q], value[m][q]) for m in o)
        )
        wins[f"{name}_best_or_tied"] = tally(
            lambda q, n=name, o=others: all(not gt(value[m][q], value[n][q]) for m in o)
        )

    # 동률이면 현재 기본값(score fusion)을 유지하고, 더 나은 방식이 있을 때만 바꾼다.
    preference = ("score_fusion", "dense", "bm25")
    choice = {q: max(preference, key=lambda m: value[m][q]) for q in query_ids}
    oracle = {
        name: fsum(rows[choice[q]][q]["metrics"][name] for q in query_ids) / count
        for name in REPORT_METRICS
    }
    means = {
        method: report["methods"][method]["slices"]["all"]
        for method in BASELINE_METHODS
    }
    oracle_block = {
        "selection_metric": metric,
        "choice_counts": dict(Counter(choice.values())),
        "metrics": oracle,
        "gain_over_dense": {n: oracle[n] - means["dense"][n] for n in REPORT_METRICS},
        "gain_over_score_fusion": {
            n: oracle[n] - means["score_fusion"][n] for n in REPORT_METRICS
        },
        "share_of_dense_to_oracle_gap_closed_by_fusion": (
            (means["score_fusion"][metric] - means["dense"][metric])
            / (oracle[metric] - means["dense"][metric])
        ),
    }

    texts = {doc.document_id: doc.text for doc in documents}
    positive = {
        q: next(d for d, rel in qrels[q].items() if rel == 3) for q in query_ids
    }
    groups = {
        "bm25_material_win": [
            q for q in query_ids if bm25[q] - dense[q] >= MATERIAL_MARGIN
        ],
        "dense_material_win": [
            q for q in query_ids if dense[q] - bm25[q] >= MATERIAL_MARGIN
        ],
        "bm25_dense_tied": [
            q for q in query_ids if abs(bm25[q] - dense[q]) <= _TIE_TOLERANCE
        ],
    }
    properties = {
        q: query_properties(rows["bm25"][q]["text"], texts[positive[q]])
        for q in query_ids
    }
    profiles = {}
    for name, ids in groups.items():
        if not ids:
            continue
        profile = {"query_count": len(ids)}
        for key in ("has_digit", "has_latin", "has_uppercase_abbreviation"):
            profile[key] = sum(properties[q][key] for q in ids) / len(ids)
        for key in ("length_chars", "bigram_overlap_with_positive"):
            profile[f"mean_{key}"] = fsum(properties[q][key] for q in ids) / len(ids)
        profile["provenance"] = dict(Counter(metadata[q]["provenance"] for q in ids))
        profile["page_type"] = dict(
            Counter(metadata[q]["page_type"] or "(none)" for q in ids)
        )
        profiles[name] = profile

    def example(q, label):
        return {
            "group": label,
            "query_id": q,
            "text": rows["bm25"][q]["text"],
            "provenance": metadata[q]["provenance"],
            "relevant": {
                d: r for d, r in sorted(qrels[q].items(), key=lambda x: -x[1])[:3]
            },
            "relevant_count": len(qrels[q]),
            metric: {m: value[m][q] for m in BASELINE_METHODS},
            "top3": {
                m: [hit["document_id"] for hit in rows[m][q]["results"][:3]]
                for m in BASELINE_METHODS
            },
            "properties": properties[q],
        }

    # 예시는 차이가 큰 순(동률이면 질의 순서)으로 기계적으로 고른다.
    def top(ids, key, n):
        return sorted(ids, key=lambda q: (-key(q), query_ids.index(q)))[:n]

    examples = [
        *(
            example(q, "bm25_clearly_better")
            for q in top(groups["bm25_material_win"], lambda q: bm25[q] - dense[q], 3)
        ),
        *(
            example(q, "dense_clearly_better")
            for q in top(groups["dense_material_win"], lambda q: dense[q] - bm25[q], 3)
        ),
        *(
            example(q, "fusion_clearly_better")
            for q in top(
                [
                    q
                    for q in query_ids
                    if fusion[q] - max(bm25[q], dense[q]) >= MATERIAL_MARGIN
                ],
                lambda q: fusion[q] - max(bm25[q], dense[q]),
                2,
            )
        ),
        *(
            example(q, "all_tied")
            for q in [
                q
                for q in query_ids
                if q in set(groups["bm25_dense_tied"])
                and abs(fusion[q] - dense[q]) <= _TIE_TOLERANCE
            ][:2]
        ),
    ]
    return {
        "split": report["dataset"]["split"],
        "metric": metric,
        "material_margin": MATERIAL_MARGIN,
        "wins": wins,
        "oracle": oracle_block,
        "lexical_profiles": profiles,
        "examples": examples,
    }

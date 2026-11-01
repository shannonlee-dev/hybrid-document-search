"""Prepare scores, select Fixed fusion, and replay frozen query encoders.

No learned fusion/router is used. Fixed fusion uses a Dense/BM25 top-100 union,
per-query union minmax and validation-selected scalar Dense weight.
"""

import json
from pathlib import Path

from data.loader import load_prepared_documents
from evaluation.data import file_sha256, load_evaluation_dataset
from evaluation.json_io import read_json, write_json
from evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from evaluation.query_encoder_training import (
    PRIMARY_PROVENANCE,
    load_encoder_inputs,
    load_query_rows,
    ranking_quality,
    training_data_audit,
)
from retrievers.base import SearchResult


def fixed_fusion_rankings(dense, sparse, alphas, *, pool_k=100, k=10):
    """Rank all scalar weights with stable corpus-order ties and union minmax."""
    import numpy as np

    dense, sparse = np.asarray(dense, dtype=float), np.asarray(sparse, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    if dense.ndim != 1 or not len(dense) or sparse.shape != dense.shape:
        raise ValueError("Dense and Sparse scores must share a nonempty corpus")
    if not np.isfinite(dense).all() or not np.isfinite(sparse).all():
        raise ValueError("scores must be finite")
    if (
        alphas.ndim != 1
        or not len(alphas)
        or not np.isfinite(alphas).all()
        or ((alphas < 0) | (alphas > 1)).any()
    ):
        raise ValueError("alphas must be a nonempty vector within [0, 1]")
    if pool_k < 1 or k < 1:
        raise ValueError("pool_k and k must be positive")
    pool = np.union1d(
        np.argsort(-dense, kind="stable")[:pool_k],
        np.argsort(-sparse, kind="stable")[:pool_k],
    )

    def normalize(values):
        return (values - values.min()) / max(float(np.ptp(values)), 1e-30)

    dd, ss = normalize(dense[pool]), normalize(sparse[pool])
    scores = alphas[:, None] * dd + (1 - alphas[:, None]) * ss
    return pool[np.argsort(-scores, axis=1, kind="stable")[:, :k]]


def select_fusion(validation_quality, alphas):
    """Freeze each encoder's alpha and the encoder using validation tables only.

    Tables have query, alpha and metric axes; metric zero is binary nDCG@10.
    Baseline is evaluated first and wins ties. Alpha ties choose the lowest alpha.
    """
    import numpy as np

    if "baseline" not in validation_quality:
        raise ValueError("baseline must be eligible for selection")
    alphas = np.asarray(alphas, dtype=float)
    if (
        alphas.ndim != 1
        or not len(alphas)
        or not np.isfinite(alphas).all()
        or ((alphas < 0) | (alphas > 1)).any()
        or (np.diff(alphas) <= 0).any()
    ):
        raise ValueError("alphas must be strictly increasing within [0, 1]")
    models = {}
    for name in ["baseline", *sorted(set(validation_quality) - {"baseline"})]:
        quality = np.asarray(validation_quality[name])
        if (
            quality.ndim != 3
            or quality.shape[1:] != (len(alphas), 3)
            or not len(quality)
            or not np.isfinite(quality).all()
        ):
            raise ValueError("quality must have finite query/alpha/three-metric axes")
        means = quality[:, :, 0].mean(axis=0)
        index = int(means.argmax())
        models[name] = {
            "alpha_index": index,
            "alpha_dense": float(alphas[index]),
            "validation_ndcg": float(means[index]),
        }
    winner = max(models, key=lambda name: models[name]["validation_ndcg"])
    return {"models": models, "selected_encoder": winner}


def _replay_sources(
    data_dir, cache_dir, score_dir, encoders, split, baseline_vectors_dir=None
):
    """List the exact inputs consumed by one replay split."""
    data_dir, cache_dir, score_dir = Path(data_dir), Path(cache_dir), Path(score_dir)
    sources = {
        "manifest": data_dir / "manifest.json",
        "corpus": data_dir / "corpus.jsonl",
        "encoding": cache_dir / "encoding.json",
        "documents": cache_dir / "document-vectors.npy",
        "score_corpus": score_dir / "corpus.json",
        "score_queries": score_dir / "queries.json",
        "dense": score_dir / "dense.npy",
        "sparse": score_dir / "sparse.npy",
        f"queries_{split}": data_dir / f"queries_{split}.jsonl",
        f"qrels_{split}": data_dir / f"qrels_{split}.jsonl",
    }
    for name, directory in sorted(encoders.items()):
        sources[f"{name}_{split}_vectors"] = (
            Path(directory) / f"selected-{split}-vectors.npy"
        )
        sources[f"{name}_checkpoint"] = Path(directory) / "selected-last-block.pt"
        sources[f"{name}_frozen"] = Path(directory) / "frozen.json"
        protocol = Path(directory) / "protocol.json"
        if protocol.is_file():
            sources[f"{name}_protocol"] = protocol
    if (score_dir / "protocol.json").is_file():
        sources["score_protocol"] = score_dir / "protocol.json"
    if baseline_vectors_dir is not None:
        directory = Path(baseline_vectors_dir)
        sources["baseline_metadata"] = directory / "baseline-vectors.json"
        sources[f"baseline_{split}_vectors"] = (
            directory / f"baseline-{split}-vectors.npy"
        )
    return sources


def prepare_scores(data_dir, cache_dir, output_dir):
    """Build independent FP64 baseline/BM25 scores from verified frozen vectors."""
    import numpy as np

    from retrievers.bm25 import BM25Retriever

    data_dir, cache_dir, output_dir = map(Path, (data_dir, cache_dir, output_dir))
    if output_dir.exists():
        raise ValueError("score output already exists; use a new directory")
    documents, vectors, queries, _ = load_encoder_inputs(
        data_dir, cache_dir, ("validation", "test")
    )
    encoding = read_json(cache_dir / "encoding.json")
    primary = [
        {**query, "split": split, "vector_row": index}
        for split, rows in queries.items()
        for index, query in enumerate(rows)
        if query["provenance"] in PRIMARY_PROVENANCE
    ]
    output_dir.mkdir(parents=True)
    dense = np.lib.format.open_memmap(
        output_dir / "dense.npy",
        mode="w+",
        dtype=np.float64,
        shape=(len(primary), len(documents)),
    )
    sparse = np.lib.format.open_memmap(
        output_dir / "sparse.npy",
        mode="w+",
        dtype=np.float64,
        shape=dense.shape,
    )
    bm25 = BM25Retriever(documents)
    document_vectors = vectors.astype(np.float64)
    row = 0
    for split in ("validation", "test"):
        path = cache_dir / f"query-vectors-{split}.npy"
        if file_sha256(path) != encoding["vector_sha256"][path.name]:
            raise ValueError("baseline query vector checksum mismatch")
        query_vectors = np.load(path, allow_pickle=False)
        if query_vectors.shape != (len(queries[split]), vectors.shape[1]):
            raise ValueError("baseline vector rows do not match queries")
        for query in [q for q in primary if q["split"] == split]:
            dense[row] = (
                query_vectors[query["vector_row"]].astype(np.float64)
                @ document_vectors.T
            )
            sparse[row] = bm25.score_all(query["text"])
            row += 1
    dense.flush()
    sparse.flush()
    write_json(output_dir / "queries.json", primary)
    write_json(
        output_dir / "corpus.json",
        [{"document_id": doc.document_id} for doc in documents],
    )
    protocol = {
        "precision": "FP32 cached vectors cast to FP64 before dot product",
        "bm25": {
            "analyzer": "char_wb",
            "ngram_range": [2, 4],
            "k1": 1.5,
            "b": 0.75,
            "method": "lucene",
        },
        "input_sha256": {
            "manifest": file_sha256(data_dir / "manifest.json"),
            "encoding": file_sha256(cache_dir / "encoding.json"),
            **encoding["vector_sha256"],
        },
        "output_sha256": {
            name: file_sha256(output_dir / name)
            for name in ("dense.npy", "sparse.npy", "queries.json", "corpus.json")
        },
    }
    write_json(output_dir / "protocol.json", protocol)
    return protocol


def _replay_tables(
    data_dir,
    cache_dir,
    score_dir,
    encoders,
    split,
    alphas,
    *,
    baseline_vectors_dir=None,
):
    import numpy as np

    documents, document_vectors, queries, qrels = load_encoder_inputs(
        data_dir, cache_dir, (split,)
    )
    score_dir = Path(score_dir)
    if [doc.document_id for doc in documents] != [
        row["document_id"] for row in read_json(score_dir / "corpus.json")
    ]:
        raise ValueError("score corpus order mismatch")
    score_queries = read_json(score_dir / "queries.json")
    lookup = {query["query_id"]: index for index, query in enumerate(score_queries)}
    if len(lookup) != len(score_queries):
        raise ValueError("score cache contains duplicate query IDs")
    dense_cache = np.load(score_dir / "dense.npy", mmap_mode="r", allow_pickle=False)
    sparse_cache = np.load(score_dir / "sparse.npy", mmap_mode="r", allow_pickle=False)
    expected_shape = (len(score_queries), len(documents))
    if dense_cache.shape != expected_shape or sparse_cache.shape != expected_shape:
        raise ValueError("score arrays do not match query/corpus identities")
    selected = [
        index
        for index, query in enumerate(queries[split])
        if query["provenance"] in PRIMARY_PROVENANCE
    ]
    if not selected:
        raise ValueError("replay requires base or faq_direct queries")
    primary_queries = [queries[split][index] for index in selected]
    missing = {query["query_id"] for query in primary_queries} - set(lookup)
    if missing:
        raise ValueError("score cache is missing evaluation queries")
    for query in primary_queries:
        cached = score_queries[lookup[query["query_id"]]]
        if cached.get("text") != query["text"] or cached.get("split", split) != split:
            raise ValueError("score query text/split mismatch")
    tables, rankings = {}, {}
    document_vectors = document_vectors.astype(np.float64)
    for name in ["baseline", *sorted(encoders)]:
        if name == "baseline":
            vectors = None
            if baseline_vectors_dir is not None:
                vectors = _load_baseline_vectors(
                    baseline_vectors_dir,
                    data_dir,
                    cache_dir,
                    split,
                    queries[split],
                    document_vectors.shape[1],
                )[selected].astype(np.float64)
        else:
            directory = Path(encoders[name])
            if (
                file_sha256(directory / "selected-last-block.pt")
                != read_json(directory / "frozen.json")["checkpoint_sha256"]
            ):
                raise ValueError("frozen checkpoint checksum mismatch")
            vectors = np.load(
                directory / f"selected-{split}-vectors.npy", allow_pickle=False
            )
            if baseline_vectors_dir is not None:
                recipe = read_json(directory / "protocol.json")
                if (
                    "BF16" not in recipe["precision"]
                    or recipe["max_eval_tokens"] != 512
                    or recipe["revision"]
                    != read_json(Path(cache_dir) / "encoding.json")["model"]["revision"]
                    or vectors.dtype != np.float32
                ):
                    raise ValueError(
                        "adapted encoder does not match BF16 encoding recipe"
                    )
            if (
                vectors.shape != (len(queries[split]), document_vectors.shape[1])
                or not np.isfinite(vectors).all()
            ):
                raise ValueError("adapted vector rows do not match split queries")
            vectors = vectors[selected].astype(np.float64)
        curve = np.zeros((len(selected), len(alphas), 3))
        ranked_rows = np.zeros(
            (len(selected), len(alphas), min(10, len(documents))), dtype=np.int32
        )
        for start in range(0, len(selected), 32):
            dense_batch = (
                None
                if vectors is None
                else vectors[start : start + 32] @ document_vectors.T
            )
            for index in range(start, min(start + 32, len(selected))):
                query_id = primary_queries[index]["query_id"]
                cache_index = lookup[query_id]
                dense = (
                    dense_cache[cache_index]
                    if dense_batch is None
                    else dense_batch[index - start]
                )
                ranked = fixed_fusion_rankings(dense, sparse_cache[cache_index], alphas)
                ranked_rows[index] = ranked
                curve[index] = [
                    ranking_quality(ranking, documents, qrels[split][query_id])
                    for ranking in ranked
                ]
        tables[name], rankings[name] = curve, ranked_rows
    return tables, rankings


def select_cached_encoders(
    data_dir,
    cache_dir,
    score_dir,
    encoders,
    output_dir,
    *,
    model_path=None,
    baseline_vectors_dir=None,
    shared_alpha=None,
):
    """Select fixed fusion and encoder on validation and write immutable choice.

    The score cache is an ordinary corpus/query score table; no judge outputs,
    learned fusion or experimental Python modules are imported.
    """
    import numpy as np

    if "baseline" in encoders:
        raise ValueError("baseline uses the original FP32 score cache")
    from evaluation.query_encoder_report import (
        encoder_provenance,
        method_configs,
        model_identity,
        source_hashes,
    )

    if not encoders:
        raise ValueError("at least one adapted encoder is required")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    alphas = np.linspace(0, 1, 101)
    tables, rankings = _replay_tables(
        data_dir,
        cache_dir,
        score_dir,
        encoders,
        "validation",
        alphas,
        baseline_vectors_dir=baseline_vectors_dir,
    )
    frozen = select_fusion(tables, alphas)
    frozen.update(
        {
            "run_id": output_dir.name,
            "alphas": alphas.tolist(),
            "selection": "validation base+faq_direct binary nDCG@10 only; baseline eligible",
            "input_sha256": {
                name: file_sha256(path)
                for name, path in _replay_sources(
                    data_dir,
                    cache_dir,
                    score_dir,
                    encoders,
                    "validation",
                    baseline_vectors_dir,
                ).items()
            },
            "precision": "baseline original FP32 score cache; adapted BF16-forward query outputs stored FP32, FP64 dot against shared fixed corpus",
        }
    )
    if shared_alpha is not None:
        frozen["shared_alpha_dense"] = shared_alpha
        method_configs(frozen)
    if baseline_vectors_dir is not None:
        frozen["precision"] = (
            "both query encoders BF16 forward, FP32 parameters/storage; shared FP32 document vectors cast to FP64 before dot product"
        )
    encoding = read_json(Path(cache_dir) / "encoding.json")
    dataset_manifest = read_json(Path(data_dir) / "manifest.json")
    protocol = {
        "schema_version": 1,
        "run_id": frozen["run_id"],
        "test_status": "previously exposed official test; exploratory reuse",
        "primary_provenance": sorted(PRIMARY_PROVENANCE),
        "fusion": "Dense/BM25 top100 union; union minmax; alpha 0:.01:1; corpus-order ties",
        "selection": frozen["selection"],
        "precision": frozen["precision"],
        "model": encoding["model"],
        "model_files_sha256": model_identity(model_path)
        if model_path is not None
        else None,
        "training_data_audit": training_data_audit(data_dir),
        "vector_sha256": encoding["vector_sha256"],
        "document_ids_sha256": encoding["document_ids_sha256"],
        "dataset_manifest_sha256": encoding["dataset_manifest_sha256"],
        "dataset": {
            key: dataset_manifest[key]
            for key in (
                "dataset_id",
                "dataset_revision",
                "datasets",
                "split_stats",
                "output_sha256",
            )
            if key in dataset_manifest
        },
        "validation_queries": len(tables["baseline"]),
        "encoders": {name: encoder_provenance(path) for name, path in encoders.items()},
        "source_sha256": source_hashes(),
    }
    if baseline_vectors_dir is not None:
        baseline_encoding = read_json(
            Path(baseline_vectors_dir) / "baseline-vectors.json"
        )
        baseline_encoding["splits"] = {
            split: {
                **{key: value for key, value in entry.items() if key != "query_ids"},
                "queries": len(entry["query_ids"]),
            }
            for split, entry in baseline_encoding["splits"].items()
        }
        protocol["baseline_encoding"] = baseline_encoding
        if (
            protocol["model_files_sha256"] is not None
            and protocol["model_files_sha256"]
            != baseline_encoding["model_files_sha256"]
        ):
            raise ValueError("matched baseline and adapted base model snapshots differ")
    write_json(output_dir / "protocol.json", protocol)
    frozen["protocol_sha256"] = file_sha256(output_dir / "protocol.json")
    np.savez_compressed(output_dir / "validation-quality.npz", **tables)
    np.savez_compressed(output_dir / "validation-rankings.npz", **rankings)
    write_json(output_dir / "frozen.json", frozen)
    return frozen


def evaluate_cached_encoders(
    data_dir, cache_dir, score_dir, encoders, output_dir, *, baseline_vectors_dir=None
):
    """Read frozen validation choices before consuming test labels or vectors."""
    import numpy as np

    output_dir = Path(output_dir)
    if any(
        (output_dir / name).exists()
        for name in (
            "per-query.jsonl",
            "summary.json",
            "test-quality.npz",
            "test-rankings.npz",
        )
    ):
        raise ValueError("evaluation output already exists; use a new run")
    frozen = read_json(output_dir / "frozen.json")
    if file_sha256(output_dir / "protocol.json") != frozen["protocol_sha256"]:
        raise ValueError("protocol changed after validation selection")
    if set(frozen["models"]) != {"baseline", *encoders}:
        raise ValueError("encoders do not match frozen validation selection")
    hashes = {
        name: file_sha256(path)
        for name, path in _replay_sources(
            data_dir, cache_dir, score_dir, encoders, "validation", baseline_vectors_dir
        ).items()
    }
    if hashes != frozen["input_sha256"]:
        raise ValueError("inputs changed after validation selection")
    tables, rankings = _replay_tables(
        data_dir,
        cache_dir,
        score_dir,
        encoders,
        "test",
        frozen["alphas"],
        baseline_vectors_dir=baseline_vectors_dir,
    )
    documents = load_prepared_documents(Path(data_dir) / "corpus.jsonl")
    dataset = load_evaluation_dataset(data_dir, "test", verify_all_splits=False)
    queries = [
        row
        for row in load_query_rows(Path(data_dir) / "queries_test.jsonl")
        if row["provenance"] in PRIMARY_PROVENANCE
    ]
    expected_ids = read_json(Path(data_dir) / "manifest.json")[
        "primary_slice_query_ids"
    ]
    if [q["query_id"] for q in queries] != expected_ids:
        raise ValueError("test queries do not match manifest primary slice")
    from evaluation.query_encoder_report import method_configs

    records = []
    for method, choice in method_configs(frozen).items():
        ranked = rankings[choice["encoder"]]
        for query, ranking in zip(
            queries, ranked[:, choice["alpha_index"]], strict=True
        ):
            ids = [documents[int(index)].document_id for index in ranking]
            hits = [SearchResult(doc, rank, 0.0) for rank, doc in enumerate(ids, 1)]
            judgments = dataset.qrels[query["query_id"]]
            records.append(
                {
                    "run_id": frozen["run_id"],
                    "query_id": query["query_id"],
                    "method": method,
                    "selection_policy": choice["selection_policy"],
                    "p@1": precision_at_k(hits, judgments, 1),
                    "recall@10": recall_at_k(hits, judgments, 10),
                    "mrr@10": reciprocal_rank_at_k(hits, judgments, 10),
                    "ndcg@10": ndcg_at_k(hits, judgments, 10),
                    "top10_document_ids": ids,
                    "relevant_rank_top10": next(
                        (
                            rank
                            for rank, doc in enumerate(ids, 1)
                            if judgments.get(doc, 0) > 0
                        ),
                        None,
                    ),
                    "selected_alpha_dense": choice["alpha_dense"],
                }
            )
    with (output_dir / "per-query.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    np.savez_compressed(output_dir / "test-quality.npz", **tables)
    np.savez_compressed(output_dir / "test-rankings.npz", **rankings)
    report = {
        "n": len(next(iter(tables.values()))),
        "selected_encoder": frozen["selected_encoder"],
        "test_status": "previously exposed official test; explicitly reused; no test selection",
        "precision": frozen["precision"],
        "models": {
            name: {
                **frozen["models"][name],
                "test": dict(
                    zip(
                        ("ndcg@10", "p@1", "recall@10"),
                        table[:, frozen["models"][name]["alpha_index"]]
                        .mean(0)
                        .tolist(),
                        strict=True,
                    )
                ),
            }
            for name, table in tables.items()
        },
        "input_sha256": {
            name: file_sha256(path)
            for name, path in _replay_sources(
                data_dir, cache_dir, score_dir, encoders, "test", baseline_vectors_dir
            ).items()
        },
    }
    for name in report["models"]:
        selected_rows = [row for row in records if row["method"] == f"{name} + Fixed"]
        metrics = {
            key: float(np.mean([row[key] for row in selected_rows]))
            for key in ("p@1", "recall@10", "mrr@10", "ndcg@10")
        }
        for key, value in report["models"][name]["test"].items():
            if not np.isclose(metrics[key], value, rtol=0, atol=1e-12):
                raise ValueError("per-query metrics disagree with replay table")
        report["models"][name]["test"] = metrics
    report["per_query"] = {
        "file": "per-query.jsonl",
        "rows": len(records),
        "sha256": file_sha256(output_dir / "per-query.jsonl"),
    }
    report["comparisons"] = {
        method: {
            metric: float(
                np.mean([row[metric] for row in records if row["method"] == method])
            )
            for metric in ("p@1", "recall@10", "mrr@10", "ndcg@10")
        }
        for method in method_configs(frozen)
    }
    report["dataset_identity"] = dataset.provenance
    write_json(output_dir / "summary.json", report)
    from evaluation.query_encoder_report import write_manifest

    write_manifest(data_dir, output_dir, records)
    return report


def _load_baseline_vectors(directory, data_dir, cache_dir, split, queries, dimension):
    """Require the exact query order, fixed corpus, BF16 recipe and vector checksum."""
    import numpy as np

    directory, data_dir, cache_dir = map(Path, (directory, data_dir, cache_dir))
    metadata = read_json(directory / "baseline-vectors.json")
    encoding = read_json(cache_dir / "encoding.json")
    recipe = metadata["query_encoding"]
    if (
        metadata["dataset_manifest_sha256"] != file_sha256(data_dir / "manifest.json")
        or metadata["document_vectors_sha256"]
        != file_sha256(cache_dir / "document-vectors.npy")
        or metadata["model"] != encoding["model"]
        or recipe["forward_dtype"] != "bfloat16"
        or recipe["storage_dtype"] != "float32"
        or recipe["max_tokens"] != 512
        or recipe["batch_size"] != 16
        or metadata["splits"][split]["query_ids"] != [q["query_id"] for q in queries]
    ):
        raise ValueError("matched baseline identity/precision/query order mismatch")
    path = directory / f"baseline-{split}-vectors.npy"
    if file_sha256(path) != metadata["splits"][split]["sha256"]:
        raise ValueError("matched baseline vector checksum mismatch")
    vectors = np.load(path, allow_pickle=False)
    if (
        vectors.shape != (len(queries), dimension)
        or vectors.dtype != np.float32
        or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
    ):
        raise ValueError("matched baseline vectors have invalid shape/dtype/values")
    return vectors

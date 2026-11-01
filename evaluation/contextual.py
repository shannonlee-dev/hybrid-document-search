"""Select DAT policy on Validation and audit canonical BC Card baselines."""

import json
from pathlib import Path

import numpy as np

from evaluation.contextual_report import compact_encoding
from evaluation.data import file_sha256, load_evaluation_dataset
from evaluation.json_io import read_json, write_json
from fusion.contextual import candidate_scores


def is_primary(query):
    return query.get("provenance") in ("base", "faq_direct")


def read_rows(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def select_vector_policy(data_directory, vector_cache, output):
    """Select the same 404 Fixed candidates directly from validation vectors."""
    from evaluation.enterprise import _load_vectors
    from retrievers.bm25 import BM25Retriever

    data_directory, vector_cache, output = (
        Path(data_directory),
        Path(vector_cache),
        Path(output),
    )
    if output.exists():
        raise FileExistsError(f"policy already exists: {output}")
    dataset = load_evaluation_dataset(
        data_directory, "validation", verify_all_splits=False
    )
    vectors, queries, encoding = _load_vectors(data_directory, vector_cache, dataset)
    metadata = read_rows(data_directory / "queries_validation.jsonl")
    primary = [i for i, q in enumerate(metadata) if is_primary(q)]
    if not primary:
        raise ValueError("non-synthetic validation queries are required")
    bm25 = BM25Retriever(dataset.documents)
    normalizations = ("full_raw", "returned_minmax", "union_minmax", "union_raw")
    totals = np.zeros((4, 101))
    for i in primary:
        query = dataset.queries[i]
        dense = np.asarray(queries[i], np.float64) @ np.asarray(vectors, np.float64).T
        sparse = bm25.score_all(query.text).astype(float)
        labels = dataset.qrels[query.query_id]
        relevant = np.array(
            [labels.get(d.document_id, 0) > 0 for d in dataset.documents]
        )
        ideal = (1 / np.log2(np.arange(min(10, int(relevant.sum()))) + 2)).sum()
        for mode, normalization in enumerate(normalizations):
            pool, dd, ss = candidate_scores(dense, sparse, normalization)
            scores = (
                np.arange(101)[:, None] / 100 * dd
                + (1 - np.arange(101)[:, None] / 100) * ss
            )
            order = np.argsort(-scores, axis=1, kind="stable")[:, :10]
            dcg = (relevant[pool[order]] / np.log2(np.arange(order.shape[1]) + 2)).sum(
                1
            )
            totals[mode] += dcg / ideal if ideal > 0 else 0
    mode, alpha_bin = np.unravel_index(totals.argmax(), totals.shape)
    selected = {
        "alpha": int(alpha_bin) / 100,
        "normalization": normalizations[mode],
        "validation_ndcg@10": float(totals[mode, alpha_bin] / len(primary)),
        "validation_queries": len(primary),
        "selection_split": "validation",
        "selection": "404 candidates; official non-synthetic validation only",
        "adaptive": "frozen local DAT odds policy; preserve top1 agreement",
        "input_sha256": {
            "dataset_manifest": dataset.provenance["manifest_sha256"],
            "encoding": file_sha256(vector_cache / "encoding.json"),
        },
        "encoding": compact_encoding(encoding),
    }
    write_json(output, selected)
    return selected


def audit_main(data_directory, vector_cache, canonical_summary, output, *, run_id):
    """Replay main's actual BM25, FAISS FP32 Dense and Fixed service contracts."""
    from evaluation.contextual_report import (
        ranking_record,
        save_handoff,
        validate_records,
    )
    from evaluation.enterprise import _build_retrievers, _load_vectors

    data_directory, output = Path(data_directory), Path(output)
    if output.exists():
        raise FileExistsError(f"run already exists: {output}")
    dataset = load_evaluation_dataset(data_directory, "test", verify_all_splits=False)
    canonical = read_json(canonical_summary)
    for name, digest in canonical["dataset"]["input_sha256"].items():
        if file_sha256(data_directory / name) != digest:
            raise ValueError(f"canonical dataset identity differs: {name}")
    vectors, qvectors, encoding = _load_vectors(data_directory, vector_cache, dataset)
    retrievers = _build_retrievers(dataset, vectors, qvectors)
    primary = read_json(data_directory / "manifest.json")["primary_slice_query_ids"]
    queries = {q.query_id: q.text for q in dataset.queries}
    indices = {d.document_id: i for i, d in enumerate(dataset.documents)}
    records = []
    for qid in primary:
        for method, (retriever, _, _) in retrievers.items():
            hits = retriever.search(queries[qid], 10)
            row = ranking_record(
                dataset,
                qid,
                [indices[h.document_id] for h in hits],
                run_id=run_id,
                method=method,
            )
            row["bm25_weight"] = 0.05 if method == "score_fusion" else None
            records.append(row)
    actual = validate_records(records, dataset, primary, list(retrievers))
    expected = {
        m: {k: canonical["methods"][m]["slices"]["non_synthetic"][k] for k in actual[m]}
        for m in retrievers
    }
    for method in actual:
        for metric in actual[method]:
            np.testing.assert_allclose(
                actual[method][metric], expected[method][metric], atol=1e-12, rtol=0
            )
    save_handoff(
        output,
        data_directory,
        records,
        selection={"selection": "unchanged main PR28 baselines; BM25 weight .05"},
        protocol={
            "scope": "main canonical baseline replay; no model fitting or new latency",
            "encoding": encoding,
            "canonical_summary_sha256": file_sha256(Path(canonical_summary)),
            "canonical_git_commit": canonical["code"]["git_commit"],
            "dense": "main DenseRetriever FAISS FP32 single-query search",
            "fixed": "main ScoreFusionRetriever; full-corpus Dense + .05*BM25/corpus-max",
        },
        baseline_audit={
            "all_three_main_baselines_reproduced": True,
            "queries": len(primary),
            "canonical_metrics": expected,
            "replayed_metrics": actual,
            "canonical_vector_sha256": canonical["encoding"]["vector_sha256"],
            "replay_vector_sha256": encoding["vector_sha256"],
            "fixed_distinction": "canonical .05 full-corpus Fixed differs from validation-selected DAT Fixed alpha=.84 union-minmax",
            "dense_distinction": "FP32 FAISS main Dense differs from float64 full-corpus NumPy DAT score path",
        },
    )

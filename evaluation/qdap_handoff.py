"""Export frozen query results and audit main baselines without fitting models."""

import hashlib
import json
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from evaluation.json_io import read_json, write_json
from evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from evaluation.qdap_benchmark import (
    METHODS,
    judgments_for,
    primary,
    rows,
    sha,
    split_scores,
)
from retrievers.base import SearchResult

METRICS = ("ndcg@10", "p@1", "recall@10", "mrr@10")


def per_query_records(
    query_ids,
    document_ids,
    rankings,
    judgments,
    *,
    run_id,
    alphas=None,
    oracle_alphas=None,
    modes=None,
    weights=None,
):
    """Join explicit query/document IDs to actual rankings and binary metrics.

    No raw text or vectors enter the handoff. A missing relevant document has
    null relevant_rank_top10. Alpha always denotes the Dense coefficient in
    union-minmax fusion; the legacy formula's BM25 weight has a separate field.
    """
    n = len(query_ids)
    if not n or len(set(query_ids)) != n or len(judgments) != n:
        raise ValueError("unique query IDs aligned with judgments required")
    if not document_ids or len(set(document_ids)) != len(document_ids):
        raise ValueError("unique document IDs required")
    alphas, modes, weights = alphas or {}, modes or {}, weights or {}
    for method, values in alphas.items():
        array = np.asarray(values, float)
        if (
            method not in rankings
            or array.shape != (n,)
            or not np.isfinite(array).all()
            or ((array < 0) | (array > 1)).any()
        ):
            raise ValueError("alpha must align with every query and method")
    if oracle_alphas is not None:
        oracle_alphas = np.asarray(oracle_alphas, float)
        if (
            oracle_alphas.shape != (n,)
            or not np.isfinite(oracle_alphas).all()
            or ((oracle_alphas < 0) | (oracle_alphas > 1)).any()
        ):
            raise ValueError("oracle alpha must align with every query")
    for mapping in (modes, weights):
        if any(
            method not in rankings or len(value) != n
            for method, value in mapping.items()
        ):
            raise ValueError("prediction metadata must align with queries")
    for ranking in rankings.values():
        ranking = np.asarray(ranking)
        if (
            ranking.ndim != 2
            or len(ranking) != n
            or not 0 < ranking.shape[1] <= 10
            or not np.issubdtype(ranking.dtype, np.integer)
        ):
            raise ValueError(
                "rankings must align with queries and contain at most ten document indices"
            )
        if (
            (ranking < 0).any()
            or (ranking >= len(document_ids)).any()
            or any(len(set(r)) != len(r) for r in ranking)
        ):
            raise ValueError("invalid or duplicate document index in ranking")
    result = []
    for i, query_id in enumerate(query_ids):
        rel = {document_ids[index]: grade for index, grade in judgments[i].items()}
        for method, ranking in rankings.items():
            ids = [document_ids[int(index)] for index in ranking[i]]
            hits = [
                SearchResult(doc_id, rank, 0.0) for rank, doc_id in enumerate(ids, 1)
            ]
            relevant_rank = next(
                (rank for rank, doc_id in enumerate(ids, 1) if rel.get(doc_id, 0) > 0),
                None,
            )
            result.append(
                {
                    "run_id": run_id,
                    "query_id": query_id,
                    "method": method,
                    "p@1": precision_at_k(hits, rel, 1),
                    "recall@10": recall_at_k(hits, rel, 10),
                    "mrr@10": reciprocal_rank_at_k(hits, rel, 10),
                    "ndcg@10": ndcg_at_k(hits, rel, 10),
                    "top10_document_ids": ids,
                    "relevant_rank_top10": relevant_rank,
                    "selected_alpha_dense": float(alphas[method][i])
                    if method in alphas
                    else None,
                    "oracle_alpha_dense": float(oracle_alphas[i])
                    if oracle_alphas is not None
                    else None,
                    "selected_mode": str(modes[method][i]) if method in modes else None,
                    "bm25_weight": float(weights[method][i])
                    if method in weights
                    else None,
                }
            )
    return result


def _averages(records):
    by_method = defaultdict(list)
    for record in records:
        by_method[record["method"]].append(record)
    return {
        name: {
            metric: float(np.mean([r[metric] for r in values])) for metric in METRICS
        }
        for name, values in by_method.items()
    }


def _write_records(path, records):
    text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in records
    )
    path.write_text(text, encoding="utf-8")


def export_frozen(args):
    """Validate the frozen run and export all 721x8 rows, including predictions."""
    from evaluation.qdap_benchmark import legacy_ranking
    from evaluation.qdap_execution import frozen_selection
    from fusion.adaptive import MODES, Router, query_features

    selection = frozen_selection(SimpleNamespace(output=args.run))
    encoding_path = (
        args.inputs / "artifacts/bccard-adaptive-vectors-resumed/encoding.json"
    )
    encoding_record = read_json(encoding_path)
    data = args.inputs / "data/processed/bccard"
    docs = rows(data / "corpus.jsonl")
    queries = read_json(args.run / "test-queries.json")
    official = [q for q in rows(data / "queries_test.jsonl") if primary(q)]
    if len(queries) != 721 or [q["query_id"] for q in queries] != [
        q["query_id"] for q in official
    ]:
        raise ValueError("saved queries must match the official ordered 721 IDs")
    labels = judgments_for(data, "test", queries, docs)
    with np.load(args.run / "test-rankings.npz", allow_pickle=False) as z:
        rankings = {method: z[method] for method in METHODS}
    with np.load(args.run / "test-predictions.npz", allow_pickle=False) as z:
        alphas = {
            "QDAP-S": z["S_alpha"],
            "QDAP-L": z["L_alpha"],
            "Oracle alpha": z["oracle_alpha"],
        }
        oracle = z["oracle_alpha"]
    alphas["Paper Fixed"] = np.full(721, selection["paper_fixed_alpha"])
    dense, sparse = split_scores(args.inputs, "test", queries)
    router = Router.load(args.inputs / "docs/experiments/adaptive/router.json")
    ids = np.array([doc["document_id"] for doc in docs])
    modes, weights = [], []
    for i, q in enumerate(queries):
        d, s = dense[i], sparse[i]
        mode = int(router.select([query_features(q["text"], s, d, ids)])[0])
        weight = router.state["weights"][mode]
        np.testing.assert_array_equal(
            legacy_ranking(d, s, ids, weight), rankings["Existing Router"][i]
        )
        modes.append(MODES[mode])
        weights.append(weight)
    records = per_query_records(
        [q["query_id"] for q in queries],
        ids.tolist(),
        rankings,
        labels,
        run_id=args.run_id,
        alphas=alphas,
        oracle_alphas=oracle,
        modes={"Existing Router": modes},
        weights={"Existing Fixed": np.full(721, 0.05), "Existing Router": weights},
    )
    expected = read_json(args.run / "test-summary.json")["rows"]
    with np.load(args.run / "test-per-query.npz", allow_pickle=False) as z:
        for method in METHODS:
            exported = [r for r in records if r["method"] == method]
            np.testing.assert_allclose(
                [[r[m] for m in METRICS] for r in exported], z[method], atol=1e-12
            )
    aggregates = _averages(records)
    for method in METHODS:
        for metric in METRICS:
            np.testing.assert_allclose(
                aggregates[method][metric], expected[method][metric], atol=1e-12
            )
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "per-query.jsonl"
    _write_records(target, records)
    write_json(
        args.output / "per-query-manifest.json",
        {
            "run_id": args.run_id,
            "queries": 721,
            "methods": list(METHODS),
            "rows": len(records),
            "file": target.name,
            "sha256": sha(target),
            "selection_sha256": sha(args.run / "selection.json"),
            "encoding": {
                key: encoding_record[key]
                for key in (
                    "model",
                    "dtype",
                    "max_seq_length",
                    "vector_sha256",
                    "dataset_manifest_sha256",
                )
            },
            "encoding_record_sha256": sha(encoding_path),
            "query_ids_sha256": hashlib.sha256(
                "\n".join(q["query_id"] for q in queries).encode()
            ).hexdigest(),
            "input_sha256": {
                str(data / name): sha(data / name)
                for name in [
                    "manifest.json",
                    "corpus.jsonl",
                    "queries_test.jsonl",
                    "qrels_test.jsonl",
                ]
            },
            "source_output_sha256": {
                name: sha(args.run / name)
                for name in [
                    "test-rankings.npz",
                    "test-per-query.npz",
                    "test-predictions.npz",
                    "test-queries.json",
                ]
            },
            "export_source_sha256": sha(__file__),
            "metrics_recomputed_and_saved_values_verified": True,
            "oracle_scope": "post-hoc nDCG argmax among101 alphas; diagnostic only",
            "aggregates": aggregates,
        },
    )
    print(
        {"exported": str(target), "rows": len(records), "bytes": target.stat().st_size},
        flush=True,
    )


def audit_main(args):
    """Replay main's unmodified CPU baseline contracts on the same frozen vectors."""
    import faiss
    from threadpoolctl import threadpool_limits

    from evaluation.data import load_evaluation_dataset
    from evaluation.enterprise import _build_retrievers, _load_vectors

    canonical = read_json(args.canonical_summary)
    data = args.inputs / "data/processed/bccard"
    for name, digest in canonical["dataset"]["input_sha256"].items():
        if sha(data / name) != digest:
            raise ValueError(f"canonical dataset identity differs: {name}")
    dataset = load_evaluation_dataset(data, "test")
    cache = args.inputs / "artifacts/bccard-adaptive-vectors-resumed"
    vectors, query_vectors, encoding = _load_vectors(data, cache, dataset)
    qs = [q for q in rows(data / "queries_test.jsonl") if primary(q)]
    ids = [d.document_id for d in dataset.documents]
    id_index = {doc_id: i for i, doc_id in enumerate(ids)}
    docs = [{"document_id": doc_id} for doc_id in ids]
    labels = judgments_for(data, "test", qs, docs)
    rankings = {}
    aliases = {"bm25": "BM25", "dense": "Dense", "score_fusion": "Existing Fixed"}
    faiss.omp_set_num_threads(2)
    with threadpool_limits(2):
        retrievers = _build_retrievers(dataset, vectors, query_vectors)
        for method, (retriever, _, _) in retrievers.items():
            rankings[aliases[method]] = np.array(
                [
                    [id_index[h.document_id] for h in retriever.search(q["text"], 10)]
                    for q in qs
                ]
            )
    records = per_query_records(
        [q["query_id"] for q in qs],
        ids,
        rankings,
        labels,
        run_id=args.main_run_id,
        weights={"Existing Fixed": np.full(721, 0.05)},
    )
    actual = _averages(records)
    for source, alias in aliases.items():
        for metric in METRICS:
            np.testing.assert_allclose(
                actual[alias][metric],
                canonical["methods"][source]["slices"]["non_synthetic"][metric],
                atol=1e-12,
            )
    saved = read_json(args.run / "test-summary.json")["rows"]
    original_dense = np.load(args.run / "test-rankings.npz")["Dense"]
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "main-baseline-per-query.jsonl"
    _write_records(target, records)
    write_json(
        args.output / "baseline-audit.json",
        {
            "run_id": args.main_run_id,
            "canonical_summary": str(args.canonical_summary),
            "canonical_summary_sha256": sha(args.canonical_summary),
            "canonical_git_commit": canonical["code"]["git_commit"],
            "queries": 721,
            "same_dataset_corpus_queries_qrels": True,
            "all_three_main_baselines_reproduced": True,
            "replayed_metrics": actual,
            "saved_qdap_run_metrics": {name: saved[name] for name in aliases.values()},
            "canonical_vector_sha256": canonical["encoding"]["vector_sha256"],
            "current_vector_sha256": encoding["vector_sha256"],
            "precision_explanation": "Using the identical current query/document vectors, main DenseRetriever's FAISS IndexFlatIP FP32 single-query path reproduces canonical metrics; the saved QDAP run used NumPy float64 score computation and document-ID ties. This is a retrieval computation/tie-order distinction, not a model or dataset change in this controlled replay.",
            "dense_top10_rankings_differ_between_paths": int(
                np.any(original_dense != rankings["Dense"], axis=1).sum()
            ),
            "existing_fixed_explanation": "main Fixed is full-corpus cosine + .05*BM25/corpus-max; Paper Fixed is separate validation-selected union-minmax alpha=.84. Do not equate their numbers.",
            "results_preserved": "Original QDAP model choices, training, test results and latency are unchanged; this audit is a separate baseline replay with no new latency measurement.",
            "per_query_file": target.name,
            "per_query_rows": len(records),
            "per_query_sha256": sha(target),
            "source_sha256": {
                str(p): sha(p)
                for p in [
                    __file__,
                    "evaluation/enterprise.py",
                    "indexing/faiss_index.py",
                    "fusion/score.py",
                    "retrievers/dense.py",
                ]
            },
        },
    )
    print(
        {
            "main_baseline_audit": "all three reproduced",
            "metrics": actual,
            "per_query_rows": len(records),
        },
        flush=True,
    )

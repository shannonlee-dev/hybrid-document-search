"""POST-HOC sensitivity of the PR #42 BC Card metrics to exact-score tie order.

Scores, candidate sets and relevance labels are those of the original run; only
the order among documents with exactly equal scores varies. Canonical orderings
are reproduced and checked against the committed PR #42 rows; they stay the
reported results. Nothing here is promoted to canonical.

Commands
  run    GPU + local data: audit duplicate vectors/labels and write per-query
         metrics for every ordering (tie-sensitivity-per-query.jsonl, IDs and
         numbers only) and duplicate-label-audit.json
  check  recompute tie-sensitivity.json from the per-query file and compare
"""

import argparse
import hashlib
import json
from math import fsum, isclose
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/validation/2026-11-02/finance-strategyqa-analysis"
SERVICE_EVIDENCE = ROOT / "docs/validation/2026-11-02/finance-service-eval"
ORDERINGS = (
    "canonical",
    "row_ascending",
    "row_descending",
    "lower_bound",
    "upper_bound",
)
METRICS = ("ndcg@10", "mrr@10", "recall@10", "p@1")
CONDITIONS = {
    "D": "Dense-only",
    "A": "service Hybrid .05 (co-primary)",
    "B": "Fixed alpha=.84 (co-primary)",
}
SEED = 20261102
RESAMPLES = 20000
TOP_K = 10


# ------------------------------------------------------------ pure helpers
def tie_blocks(scores_in_order):
    """Split a score sequence (already sorted descending) into exact-tie runs."""
    blocks, start = [], 0
    for i in range(1, len(scores_in_order) + 1):
        if i == len(scores_in_order) or scores_in_order[i] != scores_in_order[start]:
            blocks.append((start, i))
            start = i
    return blocks


def order_by(scores, rows, rule):
    """Order candidate rows by descending score with a tie rule.

    rule: 'row_ascending' or 'row_descending'. Scores are compared exactly.
    """
    sign = 1 if rule == "row_ascending" else -1
    return [
        r
        for _, _, r in sorted(zip((-s for s in scores), (sign * r for r in rows), rows))
    ]


def bound_order(ordered_rows, ordered_scores, relevant, best, k=TOP_K):
    """Rearrange each exact-tie block so relevant rows come first (best) or last.

    Block positions are fixed by the scores, so a block that crosses position k
    keeps its span; only membership inside the span changes. For binary
    relevance every reported metric is non-decreasing when a relevant document
    moves earlier, so these two arrangements attain the metric bounds.
    """
    out = []
    for start, stop in tie_blocks(ordered_scores):
        if start >= k:
            break
        block = ordered_rows[start:stop]
        rel = [r for r in block if r in relevant]
        non = [r for r in block if r not in relevant]
        out.extend(rel + non if best else non + rel)
    return out[:k]


def metrics(document_ids, judgments):
    from evaluation.metrics import (
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
        reciprocal_rank_at_k,
    )
    from retrievers.base import SearchResult

    hits = [SearchResult(d, r, 0.0) for r, d in enumerate(document_ids, 1)]
    return {
        "ndcg@10": ndcg_at_k(hits, judgments, 10),
        "mrr@10": reciprocal_rank_at_k(hits, judgments, 10),
        "recall@10": recall_at_k(hits, judgments, 10),
        "p@1": precision_at_k(hits, judgments, 1),
    }


# ------------------------------------------------------------------- run
def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def duplicate_audit(documents, vectors, queries, qrels):
    import numpy as np

    groups = {}
    for i, row in enumerate(np.ascontiguousarray(vectors, dtype=np.float32)):
        groups.setdefault(row.tobytes(), []).append(i)
    groups = [g for g in groups.values() if len(g) > 1]
    sizes = {}
    text_identical = 0
    for g in groups:
        sizes[str(len(g))] = sizes.get(str(len(g)), 0) + 1
        first = documents[g[0]]
        text_identical += all(
            documents[i].text == first.text and documents[i].title == first.title
            for i in g
        )
    conflict_queries, conflicts = set(), 0
    group_of = {i: gid for gid, g in enumerate(groups) for i in g}
    for query_id, _ in queries:
        judged = qrels[query_id]
        for g in groups:
            grades = {judged.get(documents[i].document_id, 0) > 0 for i in g}
            if len(grades) > 1:
                conflicts += 1
                conflict_queries.add(query_id)
    return (
        {
            "documents": len(documents),
            "duplicate_vector_groups": len(groups),
            "documents_in_duplicate_groups": sum(len(g) for g in groups),
            "group_size_distribution": dict(
                sorted(sizes.items(), key=lambda x: int(x[0]))
            ),
            "groups_text_and_title_identical": text_identical,
            "groups_same_vector_different_text": len(groups) - text_identical,
            "binary_label_conflicts_query_group_pairs": conflicts,
            "queries_with_binary_label_conflict": len(conflict_queries),
            "queries": len(queries),
            "group_member_rows": [g for g in groups],
        },
        group_of,
        conflict_queries,
    )


def run(args):
    import numpy as np

    from data.loader import load_prepared_documents
    from evaluation import finance_service as fs
    from fusion.minmax import UnionMinMaxFusionRetriever
    from fusion.score import DEFAULT_BM25_WEIGHT, _fuse_scores
    from indexing.faiss_index import FaissIndex
    from retrievers.adapted_query import AdaptedQueryEmbedder, load_checkpoint_identity
    from retrievers.bm25 import BM25Retriever
    from retrievers.dense import DenseConfig, DenseEmbedder, DenseRetriever

    documents = load_prepared_documents(args.data_dir / "corpus.jsonl")
    vectors = np.ascontiguousarray(
        np.load(args.cache_dir / "document-vectors.npy", allow_pickle=False),
        dtype=np.float32,
    )
    _, queries, qrels = fs.bccard_test(args.data_dir)
    audit, group_of, conflict_queries = duplicate_audit(
        documents, vectors, queries, qrels
    )
    ids = [d.document_id for d in documents]
    index = FaissIndex.build(vectors)
    bm25 = BM25Retriever(documents)
    embedders = {
        "bge": DenseEmbedder(DenseConfig(device=args.device)),
        "fin": AdaptedQueryEmbedder(
            DenseConfig(device=args.device),
            base_model_path=args.base_model,
            checkpoint_path=args.checkpoint,
            identity=load_checkpoint_identity(
                ROOT / "config/query_checkpoints/finance-seed43-fbbd5ed0.json"
            ),
        ),
    }
    committed = {}
    for row in _read_rows(SERVICE_EVIDENCE / "bccard-per-query.jsonl"):
        committed[(row["condition"], row["query_id"])] = row
    # Service Hybrid tie rule: document_id order (ScoreFusionRetriever._tie_ranks).
    tie_ranks = np.argsort(np.argsort(ids))
    rows_out, parity = [], {}
    top10_tie_blocks = {c: 0 for c in CONDITIONS}
    for encoder, embedder in embedders.items():
        dense_retriever = DenseRetriever(
            index=index, documents=documents, embedder=embedder
        )
        fixed = UnionMinMaxFusionRetriever(bm25, dense_retriever, alpha=0.84)
        for query_id, text in queries:
            q = embedder.encode_query(text)
            judged = qrels[query_id]
            relevant = {i for i, d in enumerate(ids) if judged.get(d, 0) > 0}
            sparse = np.asarray(bm25.score_all(text), dtype=np.float64)
            candidates = {}
            # D: FAISS full ordering and FAISS scores (canonical = FAISS order).
            hits = index.search(q, len(ids))
            candidates["D"] = ([r for r, _ in hits], [s for _, s in hits])
            # A: service Hybrid fused scores, canonical = document_id tie order.
            fused = _fuse_scores(
                (vectors @ q[0]).astype(np.float64),
                sparse,
                maximum=float(np.max(sparse)),
                bm25_weight=DEFAULT_BM25_WEIGHT,
            )
            order = np.lexsort((tie_ranks, -fused))
            candidates["A"] = (order.tolist(), fused[order].tolist())
            # B: Fixed alpha over the top-100 union pool, canonical = corpus order.
            dense64 = fixed.dense_scores(q[0])
            pool = np.union1d(
                np.argsort(-dense64, kind="stable")[:100],
                np.argsort(-sparse, kind="stable")[:100],
            )

            def norm(v):
                return (v - v.min()) / max(float(np.ptp(v)), 1e-30)

            pooled = 0.84 * norm(dense64[pool]) + (1 - 0.84) * norm(sparse[pool])
            border = np.argsort(-pooled, kind="stable")
            candidates["B"] = (pool[border].tolist(), pooled[border].tolist())
            for cond, (canon_rows, canon_scores) in candidates.items():
                canonical_ids = [ids[r] for r in canon_rows[:TOP_K]]
                key = (f"{cond}-{encoder}", query_id)
                parity.setdefault(f"{cond}-{encoder}", 0)
                parity[f"{cond}-{encoder}"] += (
                    canonical_ids == committed[key]["top10_document_ids"]
                )
                # Restrict re-ordering to the candidates whose scores can reach top 10.
                cutoff = canon_scores[min(TOP_K, len(canon_scores)) - 1]
                window = [i for i, s in enumerate(canon_scores) if s >= cutoff]
                w_rows = [canon_rows[i] for i in window]
                w_scores = [canon_scores[i] for i in window]
                blocks = [
                    b for b in tie_blocks(w_scores) if b[1] - b[0] > 1 and b[0] < TOP_K
                ]
                top10_tie_blocks[cond] += bool(blocks)
                orderings = {
                    "canonical": canon_rows[:TOP_K],
                    "row_ascending": order_by(w_scores, w_rows, "row_ascending")[
                        :TOP_K
                    ],
                    "row_descending": order_by(w_scores, w_rows, "row_descending")[
                        :TOP_K
                    ],
                    "lower_bound": bound_order(w_rows, w_scores, relevant, best=False),
                    "upper_bound": bound_order(w_rows, w_scores, relevant, best=True),
                }
                rows_out.append(
                    {
                        "query_id": query_id,
                        "condition": cond,
                        "encoder": encoder,
                        "top10_tie_blocks": len(blocks),
                        # some top-10 tie block holds two members of one duplicate-vector group
                        "tie_block_has_duplicate_vectors": any(
                            len(
                                g := [group_of[r] for r in w_rows[a:b] if r in group_of]
                            )
                            != len(set(g))
                            for a, b in blocks
                        ),
                        # some top-10 tie block mixes relevant and non-relevant documents:
                        # the necessary condition for tie order to change a metric
                        "top10_tie_block_label_conflict": any(
                            len({r in relevant for r in w_rows[a:b]}) > 1
                            for a, b in blocks
                        ),
                        "binary_label_conflict_query": query_id in conflict_queries,
                        **{
                            name: metrics([ids[r] for r in rows], judged)
                            for name, rows in orderings.items()
                        },
                    }
                )
    out = EVIDENCE / "tie-sensitivity-per-query.jsonl"
    with out.open("w") as stream:
        for row in rows_out:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    audit.pop("group_member_rows")
    audit["canonical_reproduction_matches_committed_top10"] = parity
    audit["queries_with_top10_exact_tie_block"] = top10_tie_blocks
    audit["source_rows_sha256"] = hashlib.sha256(
        (SERVICE_EVIDENCE / "bccard-per-query.jsonl").read_bytes()
    ).hexdigest()
    (EVIDENCE / "duplicate-label-audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    summary = summarize(EVIDENCE)
    (EVIDENCE / "tie-sensitivity.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"parity": parity, "top10_tie_blocks": top10_tie_blocks}


# --------------------------------------------------------------- summary
def _bootstrap(values_by_name, seed=SEED, resamples=RESAMPLES):
    """Joint percentile bootstrap over queries for several paired-difference vectors."""
    import numpy as np

    names = list(values_by_name)
    data = np.asarray([values_by_name[n] for n in names], dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty((resamples, len(names)))
    for start in range(0, resamples, 1000):
        stop = min(start + 1000, resamples)
        draws = rng.integers(0, data.shape[1], size=(stop - start, data.shape[1]))
        means[start:stop] = data[:, draws].mean(axis=2).T
    return {
        n: {
            "mean": float(data[i].mean()),
            "ci95": [float(v) for v in np.quantile(means[:, i], [0.025, 0.975])],
            "ci97_5_bonferroni": [
                float(v) for v in np.quantile(means[:, i], [0.0125, 0.9875])
            ],
        }
        for i, n in enumerate(names)
    }


def summarize(evidence_dir=EVIDENCE):
    rows = _read_rows(Path(evidence_dir) / "tie-sensitivity-per-query.jsonl")
    by = {}
    for r in rows:
        by.setdefault((r["condition"], r["encoder"]), {})[r["query_id"]] = r
    queries = list(by[("D", "bge")])
    result = {
        "kind": "post-hoc tie-order sensitivity; canonical PR #42 rows stay the reported results",
        "bootstrap": {
            "seed": SEED,
            "resamples": RESAMPLES,
            "joint_over": "A and B per ordering",
        },
        "orderings": {
            "canonical": "original run order (D FAISS, A document_id, B corpus order within pool)",
            "row_ascending": "exact ties by corpus row ascending",
            "row_descending": "exact ties by corpus row descending",
            "lower_bound": "relevant documents last inside every exact-tie block (per encoder)",
            "upper_bound": "relevant documents first inside every exact-tie block (per encoder)",
        },
        "conditions": {},
    }
    for cond in CONDITIONS:
        bge, fin = by[(cond, "bge")], by[(cond, "fin")]
        entry = {
            "label": CONDITIONS[cond],
            "means": {
                enc: {
                    o: {
                        m: fsum(rows_[q][o][m] for q in queries) / len(queries)
                        for m in METRICS
                    }
                    for o in ORDERINGS
                }
                for enc, rows_ in (("bge", bge), ("fin", fin))
            },
            "tie_sensitive_queries": {
                enc: sum(
                    any(
                        not isclose(
                            rows_[q][o]["ndcg@10"],
                            rows_[q]["canonical"]["ndcg@10"],
                            abs_tol=1e-12,
                        )
                        for o in ORDERINGS
                    )
                    for q in queries
                )
                for enc, rows_ in (("bge", bge), ("fin", fin))
            },
        }
        diffs = {
            o: [fin[q][o]["ndcg@10"] - bge[q][o]["ndcg@10"] for q in queries]
            for o in ("canonical", "row_ascending", "row_descending")
        }
        # Worst / best case for the Finance advantage under independent tie orders.
        diffs["finance_worst_case"] = [
            fin[q]["lower_bound"]["ndcg@10"] - bge[q]["upper_bound"]["ndcg@10"]
            for q in queries
        ]
        diffs["finance_best_case"] = [
            fin[q]["upper_bound"]["ndcg@10"] - bge[q]["lower_bound"]["ndcg@10"]
            for q in queries
        ]
        entry["paired_ndcg_difference"] = diffs
        result["conditions"][cond] = entry
    # Joint bootstrap of A and B per ordering (same resample indices), as in the addendum.
    for ordering in (
        "canonical",
        "row_ascending",
        "row_descending",
        "finance_worst_case",
        "finance_best_case",
    ):
        boot = _bootstrap(
            {
                c: result["conditions"][c]["paired_ndcg_difference"][ordering]
                for c in ("A", "B")
            }
        )
        for c in ("A", "B"):
            result["conditions"][c].setdefault("paired_bootstrap", {})[ordering] = boot[
                c
            ]
        d = _bootstrap(
            {"D": result["conditions"]["D"]["paired_ndcg_difference"][ordering]}
        )["D"]
        result["conditions"]["D"].setdefault("paired_bootstrap", {})[ordering] = d
    for c in CONDITIONS:
        result["conditions"][c].pop("paired_ndcg_difference")
    return result


def _close(a, b, rel=1e-9):
    if isinstance(a, float) and isinstance(b, float):
        return isclose(a, b, rel_tol=rel, abs_tol=1e-12)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], rel) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, rel) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def check(evidence_dir=EVIDENCE):
    stored = json.loads((Path(evidence_dir) / "tie-sensitivity.json").read_text())
    if not _close(json.loads(json.dumps(summarize(evidence_dir))), stored):
        raise ValueError("tie-sensitivity.json does not match recomputation")
    return {"verified": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("run", "check"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = run(args) if args.command == "run" else check()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

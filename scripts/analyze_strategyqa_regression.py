"""Exploratory analysis of the Finance encoder's Ko-StrategyQA regression.

Commands
  features  compute per-query features for Ko-StrategyQA and BC Card Test with
            the service FP32 query encoders and frozen document vectors
            (needs GPU and local data; writes per-query-summary.jsonl, no text)
  cases     print the largest Dense losses and gains with text to stdout only,
            for manual review (nothing is written)
  analyze   statistics, figures and MANIFEST from committed files only
  check     recompute `analyze` and verify MANIFEST.sha256

See docs/validation/2026-11-02/finance-strategyqa-analysis/preregistration.md.
Exploratory: associations here generate hypotheses; they do not prove causes.
"""

import argparse
import hashlib
import json
from math import isclose
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/validation/2026-11-02/finance-strategyqa-analysis"
SERVICE_EVIDENCE = ROOT / "docs/validation/2026-11-02/finance-service-eval"
IDENTITY = ROOT / "config/query_checkpoints/finance-seed43-fbbd5ed0.json"
FINANCE_TERMS = (
    "카드", "은행", "대출", "이자", "금리", "결제", "보험", "주식", "투자", "세금",
    "연회비", "포인트", "계좌", "수수료", "적금", "예금", "환율", "신용", "할부",
    "한도", "금융", "돈", "가격", "비용",
)  # fmt: skip
CONDITIONS = {"D": ("D-fin", "D-bge"), "A": ("A-fin", "A-bge"), "B": ("B-fin", "B-bge")}
DATASETS = {
    "strategyqa": "strategyqa-per-query.jsonl",
    "bccard": "bccard-per-query.jsonl",
}
SEED = 20261102
RESAMPLES = 5000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def char_bigrams(text):
    compact = "".join(text.split())
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def lexical_overlap(query, documents):
    q = char_bigrams(query)
    best = 0.0
    for text in documents:
        d = char_bigrams(text)
        if q and d:
            best = max(best, len(q & d) / len(q | d))
    return best


# ------------------------------------------------------------------ features
def _load_dataset(name, args):
    """Return documents, document vectors, (query_id, text) pairs and qrels."""
    import numpy as np

    from data.loader import load_prepared_documents
    from evaluation import finance_service as fs

    if name == "strategyqa":
        _, documents, vectors, _, queries, qrels = fs.load_external(
            args.external_cache, "strategyqa"
        )
        return documents, vectors, queries, qrels
    documents = load_prepared_documents(args.data_dir / "corpus.jsonl")
    vectors = np.load(args.cache_dir / "document-vectors.npy", allow_pickle=False)
    _, queries, qrels = fs.bccard_test(args.data_dir)
    return documents, vectors, queries, qrels


def _embedders(args):
    from retrievers.adapted_query import AdaptedQueryEmbedder, load_checkpoint_identity
    from retrievers.dense import DenseConfig, DenseEmbedder

    bge = DenseEmbedder(DenseConfig(device=args.device))
    fin = AdaptedQueryEmbedder(
        DenseConfig(device=args.device),
        base_model_path=args.base_model,
        checkpoint_path=args.checkpoint,
        identity=load_checkpoint_identity(IDENTITY),
    )
    bge.encode_query("warm-up")
    fin.encode_query("warm-up")
    return {"bge": bge, "fin": fin}


def _prenorm_norm(embedder, text):
    import numpy as np

    vector = embedder.model.encode(
        [text], normalize_embeddings=False, convert_to_numpy=True, prompt=""
    )
    return float(np.linalg.norm(vector[0]))


def compute_features(args):
    import numpy as np

    from data.preprocess import build_index_text

    embedders = _embedders(args)
    tokenizer = embedders["bge"].model.tokenizer
    bccard_documents, bccard_vectors, *_ = _load_dataset("bccard", args)
    finance_centroid = bccard_vectors.astype(np.float64).mean(0)
    finance_centroid /= np.linalg.norm(finance_centroid)
    summary, parity = [], {}
    for dataset, rows_file in DATASETS.items():
        documents, vectors, queries, qrels = _load_dataset(dataset, args)
        own_centroid = vectors.astype(np.float64).mean(0)
        own_centroid /= np.linalg.norm(own_centroid)
        index_of = {doc.document_id: i for i, doc in enumerate(documents)}
        committed = {}
        for row in read_rows(SERVICE_EVIDENCE / rows_file):
            committed.setdefault(row["condition"], {})[row["query_id"]] = row
        # v2: rank features come from the canonical service path (FAISS
        # IndexFlatIP over the same FP32 vectors, full ordering), so exact ties
        # between duplicate document vectors resolve exactly as in PR #42.
        # Scores for margins/gaps use NumPy FP32 (max/order statistics are
        # tie-invariant; FAISS vs NumPy score error <= 3.6e-7, ranking-parity.json).
        from indexing.faiss_index import FaissIndex

        index = FaissIndex.build(np.ascontiguousarray(vectors, dtype=np.float32))
        mismatches = {"bge": 0, "fin": 0}
        for query_id, text in queries:
            relevant = [index_of[d] for d, g in qrels[query_id].items() if g > 0]
            if not relevant:
                raise ValueError(f"query without relevant documents: {query_id}")
            row = {"dataset": dataset, "query_id": query_id}
            scores, top100, vec = {}, {}, {}
            for name, embedder in embedders.items():
                q = embedder.encode_query(text)[0]
                vec[name] = q.astype(np.float64)
                s = vectors @ q
                order = np.asarray(
                    [r for r, _ in index.search(q[None, :], len(documents))]
                )
                position = np.empty(len(order), dtype=np.int64)
                position[order] = np.arange(1, len(order) + 1)
                relevant_mask = np.zeros(len(s), dtype=bool)
                relevant_mask[relevant] = True
                best_rel = float(s[relevant_mask].max())
                nonrel = np.sort(s[~relevant_mask])[::-1]
                best_non = float(nonrel[0])
                z = (s.astype(np.float64) - s.mean()) / s.std()
                z_rel = float(z[relevant_mask].max())
                z_non = np.sort(z[~relevant_mask])[::-1]
                top10 = [documents[i].document_id for i in order[:10]]
                if top10 != committed[f"D-{name}"][query_id]["top10_document_ids"]:
                    mismatches[name] += 1
                scores[name] = s
                top100[name] = set(order[:100].tolist())
                row[f"first_relevant_rank_{name}"] = int(position[relevant].min())
                row[f"best_relevant_score_{name}"] = best_rel
                row[f"best_nonrelevant_score_{name}"] = best_non
                row[f"margin_{name}"] = best_rel - best_non
                row[f"corpus_score_mean_{name}"] = float(s.mean())
                row[f"corpus_score_std_{name}"] = float(s.std())
                for k in (1, 5, 10):
                    # best relevant score minus the k-th best non-relevant score
                    row[f"gap_k{k}_{name}"] = best_rel - float(nonrel[k - 1])
                    row[f"zgap_k{k}_{name}"] = z_rel - float(z_non[k - 1])
                row[f"prenorm_norm_{name}"] = _prenorm_norm(embedder, text)
                row[f"finance_centroid_cos_{name}"] = float(
                    vec[name] @ finance_centroid
                )
                row[f"own_centroid_cos_{name}"] = float(vec[name] @ own_centroid)
                row[f"top10_{name}"] = top10
            row["query_cosine_bge_fin"] = float(
                vec["bge"]
                @ vec["fin"]
                / (np.linalg.norm(vec["bge"]) * np.linalg.norm(vec["fin"]))
            )
            row["top10_overlap"] = (
                len(set(row.pop("top10_bge")) & set(row.pop("top10_fin"))) / 10
            )
            row["top100_overlap"] = len(top100["bge"] & top100["fin"]) / 100
            row["score_correlation_bge_fin"] = float(
                np.corrcoef(scores["bge"], scores["fin"])[0, 1]
            )
            row["char_length"] = len(text)
            row["token_length"] = len(
                tokenizer(text, add_special_tokens=False)["input_ids"]
            )
            row["relevant_count"] = len(relevant)
            row["finance_term"] = any(term in text for term in FINANCE_TERMS)
            row["lexical_overlap"] = lexical_overlap(
                text, [build_index_text(documents[i]) for i in relevant]
            )
            for cond, (treated, control) in CONDITIONS.items():
                for metric in ("ndcg@10", "mrr@10", "recall@10"):
                    row[f"delta_{metric}_{cond}"] = (
                        committed[treated][query_id][metric]
                        - committed[control][query_id][metric]
                    )
            summary.append(row)
        parity[dataset] = {
            "queries": len(queries),
            "dense_top10_mismatches": mismatches,
        }
    out = EVIDENCE / "per-query-summary.jsonl"
    with out.open("w", encoding="utf-8") as stream:
        for row in summary:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    (EVIDENCE / "features-parity.json").write_text(
        json.dumps(
            {
                "check": "Dense top10 from features (v2: FAISS full ordering) vs committed PR #42 D-bge/D-fin rows",
                "datasets": parity,
                "identity_sha256": sha256(IDENTITY),
            },
            indent=2,
        )
        + "\n"
    )
    return parity


def print_cases(args):
    """Print largest Dense losses/gains with text for manual review only."""
    rows = [
        r
        for r in read_rows(EVIDENCE / "per-query-summary.jsonl")
        if r["dataset"] == "strategyqa"
    ]
    _, _, queries, qrels = _load_dataset("strategyqa", args)
    data = read_json(args.external_cache / "strategyqa" / "data.json")
    titles = {
        d["document_id"]: (d.get("title"), d["text"][:160]) for d in data["documents"]
    }
    text = dict(queries)
    rows.sort(key=lambda r: r["delta_ndcg@10_D"])
    for label, chosen in (("LOSS", rows[:15]), ("GAIN", rows[-5:][::-1])):
        for r in chosen:
            print(
                f"[{label}] {r['query_id']} dN={r['delta_ndcg@10_D']:+.3f} rank {r['first_relevant_rank_bge']}->{r['first_relevant_rank_fin']} cos={r['query_cosine_bge_fin']:.4f} overlap={r['lexical_overlap']:.3f} rel={r['relevant_count']} fin_term={r['finance_term']}"
            )
            print("   Q:", text[r["query_id"]])
            for d, g in qrels[r["query_id"]].items():
                if g > 0:
                    print("   gold:", titles[d][0], "|", titles[d][1])


def _first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def _ndcg_ids(ids, judgments):
    from evaluation.metrics import ndcg_at_k
    from retrievers.base import SearchResult

    hits = [SearchResult(d, r, 0.0) for r, d in enumerate(ids, 1)]
    return ndcg_at_k(hits, judgments, 10)


def _relevant_rank(ranking, judgments):
    return next((i for i, d in enumerate(ranking, 1) if judgments.get(d, 0) > 0), None)


def ranking_parity(args):
    """Compare canonical FAISS service rankings with NumPy rankings per query.

    For every query and encoder: canonical committed top10 (service FAISS path,
    PR #42), FAISS recomputed now, NumPy FP32 and NumPy FP64 with stable
    corpus-order ties. Mismatches are attributed to exact score ties between
    bitwise-identical document vectors, or to real score differences.
    """
    import numpy as np

    from indexing.faiss_index import FaissIndex

    embedders = _embedders(args)
    report = {
        "kind": "ranking parity investigation (follow-up review of PR #43)",
        "ranking_paths": {
            "canonical": "PR #42 committed D-* rows: DenseRetriever.search -> FAISS IndexFlatIP FP32, k=10",
            "faiss_now": "FaissIndex.build(document vectors).search(query vector, 100)",
            "numpy_fp32": "document_vectors_fp32 @ query_fp32, argsort(kind=stable)",
            "numpy_fp64": "document_vectors_fp64 @ query_fp64, argsort(kind=stable)",
        },
        "datasets": {},
    }
    for dataset, rows_file in DATASETS.items():
        documents, vectors, queries, qrels = _load_dataset(dataset, args)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        vectors64 = vectors.astype(np.float64)
        index = FaissIndex.build(vectors)
        digest = {}
        for i, row in enumerate(vectors):
            digest.setdefault(row.tobytes(), []).append(i)
        groups = [g for g in digest.values() if len(g) > 1]
        group_of = np.full(len(vectors), -1)
        for gid, g in enumerate(groups):
            group_of[g] = gid
        committed = {}
        for row in read_rows(SERVICE_EVIDENCE / rows_file):
            committed.setdefault(row["condition"], {})[row["query_id"]] = row
        ids = [doc.document_id for doc in documents]
        row_of = {d: i for i, d in enumerate(ids)}
        summary = {
            "documents": len(vectors),
            "vectors_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
            "duplicate_vector_groups": len(groups),
            "documents_in_duplicate_groups": int((group_of >= 0).sum()),
            "max_abs_document_norm_error": float(
                np.abs(np.linalg.norm(vectors64, axis=1) - 1).max()
            ),
        }
        cases, totals = [], {}
        for encoder, embedder in embedders.items():
            t = {
                "queries": 0,
                "query_vector_reencode_bitwise_equal": 0,
                "faiss_now_equals_canonical": 0,
                "numpy32_equals_canonical": 0,
                "numpy64_equals_numpy32": 0,
                "mismatch_same_top10_set": 0,
                "mismatch_different_top10_set": 0,
                "mismatch_all_differences_exact_ties_duplicate_vectors": 0,
                "mismatch_other": 0,
                "mismatch_ndcg_changed": 0,
                "max_abs_score_error_faiss_vs_numpy32": 0.0,
            }
            for query_id, text in queries:
                q = embedder.encode_query(text)
                t["queries"] += 1
                t["query_vector_reencode_bitwise_equal"] += bool(
                    np.array_equal(q, embedder.encode_query(text))
                )
                canonical = committed[f"D-{encoder}"][query_id]["top10_document_ids"]
                hits = index.search(q, 100)
                faiss_ids = [ids[r] for r, _ in hits]
                s32 = vectors @ q[0]
                s64 = vectors64 @ q[0].astype(np.float64)
                np32 = [ids[r] for r in np.argsort(-s32, kind="stable")[:10]]
                np64 = [ids[r] for r in np.argsort(-s64, kind="stable")[:10]]
                error = max(abs(score - float(s32[r])) for r, score in hits)
                t["max_abs_score_error_faiss_vs_numpy32"] = max(
                    t["max_abs_score_error_faiss_vs_numpy32"], error
                )
                t["faiss_now_equals_canonical"] += faiss_ids[:10] == canonical
                t["numpy64_equals_numpy32"] += np64 == np32
                if np32 == canonical:
                    t["numpy32_equals_canonical"] += 1
                    continue
                differing = [
                    (row_of[a], row_of[b]) for a, b in zip(canonical, np32) if a != b
                ]
                ties = all(
                    s32[a] == s32[b] and group_of[a] >= 0 and group_of[a] == group_of[b]
                    for a, b in differing
                )
                same_set = set(canonical) == set(np32)
                judgments = qrels[query_id]
                ndcg_c = _ndcg_ids(canonical, judgments)
                ndcg_n = _ndcg_ids(np32, judgments)
                t[
                    "mismatch_same_top10_set"
                    if same_set
                    else "mismatch_different_top10_set"
                ] += 1
                t[
                    "mismatch_all_differences_exact_ties_duplicate_vectors"
                    if ties
                    else "mismatch_other"
                ] += 1
                t["mismatch_ndcg_changed"] += abs(ndcg_c - ndcg_n) > 1e-12
                a, b = differing[0]
                cases.append(
                    {
                        "encoder": encoder,
                        "query_id": query_id,
                        "first_difference_position": _first_difference(canonical, np32)
                        + 1,
                        "top10_set_equal": same_set,
                        "positions_agreeing": sum(
                            x == y for x, y in zip(canonical, np32)
                        ),
                        "relevant_rank_canonical": _relevant_rank(canonical, judgments),
                        "relevant_rank_numpy32": _relevant_rank(np32, judgments),
                        "ndcg_canonical": ndcg_c,
                        "ndcg_numpy32": ndcg_n,
                        "first_pair_score_gap_numpy32": float(s32[a] - s32[b]),
                        "first_pair_same_vector_group": bool(
                            group_of[a] >= 0 and group_of[a] == group_of[b]
                        ),
                        "first_pair_same_text": documents[a].text == documents[b].text,
                        "first_pair_corpus_rows": [int(a), int(b)],
                        "all_differences_exact_ties_duplicate_vectors": ties,
                        "faiss_now_equals_canonical": faiss_ids[:10] == canonical,
                        "numpy64_equals_numpy32": np64 == np32,
                    }
                )
            totals[encoder] = t
        report["datasets"][dataset] = {
            **summary,
            "encoders": totals,
            "mismatch_cases": cases,
        }
    (EVIDENCE / "ranking-parity.json").write_text(json.dumps(report, indent=2) + "\n")
    return {d: v["encoders"] for d, v in report["datasets"].items()}


def _source(dataset, document_id):
    if dataset == "strategyqa":
        return document_id.rpartition("-")[0]  # Wikipedia title; IDs are title-index
    from data.bccard import parent_and_index

    return parent_and_index(document_id)[0]


def posthoc_same_source(args):
    """POST-HOC (not preregistered): is a newly top-1 non-relevant doc same-source?

    Uses committed Dense rows and local qrels; writes counts only.
    """
    result = {
        "kind": "post-hoc, added after manual case review; not preregistered",
        "source_definition": {
            "strategyqa": "Wikipedia title (document_id = title-index)",
            "bccard": "parent document (data.bccard.parent_and_index)",
        },
        "datasets": {},
    }
    for dataset, rows_file in DATASETS.items():
        *_, qrels = _load_dataset(dataset, args)
        rows = {}
        for row in read_rows(SERVICE_EVIDENCE / rows_file):
            rows.setdefault(row["condition"], {})[row["query_id"]] = row
        stats = {}
        for name, (control, treated) in {
            "lost_rank1": ("D-bge", "D-fin"),
            "gained_rank1": ("D-fin", "D-bge"),
        }.items():
            n = same = 0
            for query_id, row in rows[treated].items():
                gold = {d for d, g in qrels[query_id].items() if g > 0}
                if rows[control][query_id]["top10_document_ids"][0] in gold and (
                    row["top10_document_ids"][0] not in gold
                ):
                    n += 1
                    sources = {_source(dataset, d) for d in gold}
                    same += _source(dataset, row["top10_document_ids"][0]) in sources
            stats[name] = {"queries": n, "new_top1_same_source_as_gold": same}
        for encoder in ("bge", "fin"):
            n = same = 0
            for query_id, row in rows[f"D-{encoder}"].items():
                gold = {d for d, g in qrels[query_id].items() if g > 0}
                top1 = row["top10_document_ids"][0]
                if top1 not in gold:
                    n += 1
                    same += _source(dataset, top1) in {
                        _source(dataset, d) for d in gold
                    }
            stats[f"all_nonrelevant_top1_{encoder}"] = {
                "queries": n,
                "same_source_as_gold": same,
            }
        result["datasets"][dataset] = stats
    (EVIDENCE / "posthoc-same-source.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def write_v1_records(evidence_dir=EVIDENCE):
    """Freeze the v1->v2 comparison and the v1 file provenance (needs git history)."""
    evidence_dir = Path(evidence_dir)
    revisions = {
        "kind": "v1 (NumPy stable tie order) files removed from the tree; kept in git history",
        "v1_commit": V1_COMMIT,
        "renamed_copy_commit": "73a9c8f0bbf0d1ce27e921ca2ede7863c4194b5c",
        "reason_removed": "superseded by canonical-tie v2; differences frozen in v1-v2-comparison.json",
        "files": {
            name: {
                "path_at_v1_commit": _evidence_path(name),
                "path_at_renamed_copy_commit": _evidence_path(
                    name.rsplit(".", 1)[0] + ".v1-numpy-ties." + name.rsplit(".", 1)[1]
                ),
                "sha256": hashlib.sha256(
                    _git_show(V1_COMMIT, _evidence_path(name))
                ).hexdigest(),
            }
            for name in V1_FILES
        },
    }
    (evidence_dir / "revisions.json").write_text(json.dumps(revisions, indent=2) + "\n")
    comparison = {
        "v1_source": f"{V1_COMMIT}:{_evidence_path('per-query-summary.jsonl')}",
        "v2_source": "per-query-summary.jsonl",
        "comparison": v1_v2_differences(evidence_dir),
    }
    (evidence_dir / "v1-v2-comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )
    return comparison["comparison"]


# ------------------------------------------------------------------- analyze
def _bootstrap_mean(values, seed=SEED, resamples=RESAMPLES):
    import numpy as np

    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "ci95": [float(low), float(high)],
    }


def _spearman(x, y):
    from scipy.stats import spearmanr

    result = spearmanr(x, y)
    return {
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "n": len(x),
    }


def _describe(values):
    import numpy as np

    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p10": float(np.quantile(v, 0.1)),
        "p90": float(np.quantile(v, 0.9)),
    }


def score_scale_vs_separation(rs):
    """POST-HOC review: separate a uniform score shift from ranking separation.

    All statistics are per query over the full corpus (same document set for
    both encoders). gap_kK = best relevant - K-th best non-relevant score; its
    sign says whether the best relevant document beats the K-th non-relevant
    one. zgap uses per-query corpus z-scores, removing scale and location.
    """
    import numpy as np

    out = {
        "scope": "per query, full corpus; same documents for both encoders",
        "delta_corpus_score_mean": _describe(
            [r["corpus_score_mean_fin"] - r["corpus_score_mean_bge"] for r in rs]
        ),
        "delta_corpus_score_std": _describe(
            [r["corpus_score_std_fin"] - r["corpus_score_std_bge"] for r in rs]
        ),
    }
    for k in (1, 5, 10):
        raw = [r[f"gap_k{k}_fin"] - r[f"gap_k{k}_bge"] for r in rs]
        z = [r[f"zgap_k{k}_fin"] - r[f"zgap_k{k}_bge"] for r in rs]
        out[f"k{k}"] = {
            "delta_gap_raw": _describe(raw),
            "delta_gap_z": _describe(z),
            "share_delta_gap_z_negative": float(np.mean(np.asarray(z) < 0)),
            "relevant_above_kth_nonrelevant_lost": sum(
                r[f"gap_k{k}_bge"] > 0 >= r[f"gap_k{k}_fin"] for r in rs
            ),
            "relevant_above_kth_nonrelevant_gained": sum(
                r[f"gap_k{k}_bge"] <= 0 < r[f"gap_k{k}_fin"] for r in rs
            ),
            "spearman_delta_gap_z_vs_dense_delta_ndcg": _spearman(
                z, [r["delta_ndcg@10_D"] for r in rs]
            ),
        }
    return out


def overlap_outlier_check(rs):
    """POST-HOC review: is the low-overlap loss driven by a few large losses?"""
    import numpy as np
    from scipy.stats import trim_mean

    t = _tertiles(rs, "lexical_overlap")
    out = {"cuts": t["cuts"]}
    for name, group in t["groups"].items():
        d = np.sort(np.asarray([r["delta_ndcg@10_D"] for r in group]))
        out[name] = {
            "n": int(len(d)),
            "mean": float(d.mean()),
            "median": float(np.median(d)),
            "trimmed_mean_10pct": float(trim_mean(d, 0.1)),
            "mean_without_5_largest_losses": float(d[5:].mean()),
            "share_losses": float(np.mean(d < -1e-12)),
            "share_gains": float(np.mean(d > 1e-12)),
            "sum_of_5_largest_losses": float(d[:5].sum()),
        }
    return out


V1_COMMIT = "19d9d147ed0e0c87140c7a1fbd1fd500cb76e590"
V1_FILES = (
    "per-query-summary.jsonl",
    "analysis.json",
    "features-parity.json",
    "rank-movement.svg",
    "drift-vs-delta.svg",
)


def _git_show(commit, path):
    import subprocess

    return subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, check=True
    ).stdout


def _evidence_path(name):
    return f"{EVIDENCE.relative_to(ROOT).as_posix()}/{name}"


def v1_v2_differences(evidence_dir):
    """Record how rank-based features changed between v1 (NumPy ties) and v2.

    v1 is read from git history (commit V1_COMMIT), not from the working tree.
    """
    v1_text = _git_show(V1_COMMIT, _evidence_path("per-query-summary.jsonl")).decode()
    v1 = {
        (r["dataset"], r["query_id"]): r
        for r in (json.loads(line) for line in v1_text.splitlines())
    }
    v2 = {
        (r["dataset"], r["query_id"]): r
        for r in read_rows(evidence_dir / "per-query-summary.jsonl")
    }
    fields = (
        "first_relevant_rank_bge",
        "first_relevant_rank_fin",
        "top10_overlap",
        "top100_overlap",
        "margin_bge",
        "margin_fin",
        "query_cosine_bge_fin",
        "delta_ndcg@10_D",
    )
    out = {}
    for dataset in DATASETS:
        keys = [k for k in v2 if k[0] == dataset]
        changed = {
            f: sum(
                not isclose(float(v1[k][f]), float(v2[k][f]), rel_tol=0, abs_tol=1e-9)
                for k in keys
            )
            for f in fields
        }
        out[dataset] = {
            "queries": len(keys),
            "queries_with_changed_field": changed,
            "dense_rank_movement_v1": movement([v1[k] for k in keys]),
            "dense_rank_movement_v2": movement([v2[k] for k in keys]),
        }
    return out


def _tertiles(rows, key):
    import numpy as np

    values = np.asarray([r[key] for r in rows], dtype=float)
    cuts = np.quantile(values, [1 / 3, 2 / 3])
    groups = {"low": [], "mid": [], "high": []}
    for r, v in zip(rows, values, strict=True):
        groups["low" if v <= cuts[0] else "mid" if v <= cuts[1] else "high"].append(r)
    return {"cuts": [float(c) for c in cuts], "groups": groups}


def movement(rows, cond="D"):
    """Classify first-relevant rank movement for the Dense condition."""
    counts = {
        "entered_top10": 0,
        "left_top10": 0,
        "up_in_top10": 0,
        "down_in_top10": 0,
        "same_in_top10": 0,
        "outside_both": 0,
    }
    for r in rows:
        b, f = r["first_relevant_rank_bge"], r["first_relevant_rank_fin"]
        if b > 10 and f <= 10:
            counts["entered_top10"] += 1
        elif b <= 10 and f > 10:
            counts["left_top10"] += 1
        elif b <= 10 and f <= 10:
            counts[
                "up_in_top10"
                if f < b
                else "down_in_top10"
                if f > b
                else "same_in_top10"
            ] += 1
        else:
            counts["outside_both"] += 1
    return counts


def analyze(evidence_dir=EVIDENCE):
    import numpy as np

    evidence_dir = Path(evidence_dir)
    rows = read_rows(evidence_dir / "per-query-summary.jsonl")
    out = {
        "kind": "exploratory; associations are hypothesis-generating, not causal",
        "inputs_sha256": {
            name: sha256(evidence_dir / name)
            for name in (
                "per-query-summary.jsonl",
                "features-parity.json",
                "ranking-parity.json",
                "v1-v2-comparison.json",
                "revisions.json",
                "manual-review.json",
                "posthoc-same-source.json",
            )
        },
        "datasets": {},
    }
    for dataset in DATASETS:
        rs = [r for r in rows if r["dataset"] == dataset]
        d = {"queries": len(rs)}
        d["wins_ties_losses"] = {
            cond: [
                sum(r[f"delta_ndcg@10_{cond}"] > 1e-12 for r in rs),
                sum(abs(r[f"delta_ndcg@10_{cond}"]) <= 1e-12 for r in rs),
                sum(r[f"delta_ndcg@10_{cond}"] < -1e-12 for r in rs),
            ]
            for cond in CONDITIONS
        }
        d["mean_deltas"] = {
            cond: {
                m: float(np.mean([r[f"delta_{m}_{cond}"] for r in rs]))
                for m in ("ndcg@10", "mrr@10", "recall@10")
            }
            for cond in CONDITIONS
        }
        d["dense_rank_movement"] = movement(rs)
        log_ratio = [
            np.log2(r["first_relevant_rank_fin"] / r["first_relevant_rank_bge"])
            for r in rs
        ]
        d["dense_first_relevant_log2_rank_ratio"] = _describe(log_ratio)
        drift = [1 - r["query_cosine_bge_fin"] for r in rs]
        d["embedding"] = {
            "query_drift_1_minus_cos": _describe(drift),
            "prenorm_norm_bge": _describe([r["prenorm_norm_bge"] for r in rs]),
            "prenorm_norm_fin": _describe([r["prenorm_norm_fin"] for r in rs]),
            "delta_best_relevant_score": _describe(
                [
                    r["best_relevant_score_fin"] - r["best_relevant_score_bge"]
                    for r in rs
                ]
            ),
            "delta_best_nonrelevant_score": _describe(
                [
                    r["best_nonrelevant_score_fin"] - r["best_nonrelevant_score_bge"]
                    for r in rs
                ]
            ),
            "delta_margin": _describe([r["margin_fin"] - r["margin_bge"] for r in rs]),
            "top10_overlap": _describe([r["top10_overlap"] for r in rs]),
            "top100_overlap": _describe([r["top100_overlap"] for r in rs]),
            "score_correlation_bge_fin": _describe(
                [r["score_correlation_bge_fin"] for r in rs]
            ),
            "delta_finance_centroid_cos": _describe(
                [
                    r["finance_centroid_cos_fin"] - r["finance_centroid_cos_bge"]
                    for r in rs
                ]
            ),
            "delta_own_centroid_cos": _describe(
                [r["own_centroid_cos_fin"] - r["own_centroid_cos_bge"] for r in rs]
            ),
        }
        dn = [r["delta_ndcg@10_D"] for r in rs]
        d["associations_with_dense_delta_ndcg"] = {
            "H1_drift": _spearman(drift, dn),
            "H2_delta_margin": _spearman(
                [r["margin_fin"] - r["margin_bge"] for r in rs], dn
            ),
            "H3_delta_finance_centroid_cos": _spearman(
                [
                    r["finance_centroid_cos_fin"] - r["finance_centroid_cos_bge"]
                    for r in rs
                ],
                dn,
            ),
            "delta_own_centroid_cos": _spearman(
                [r["own_centroid_cos_fin"] - r["own_centroid_cos_bge"] for r in rs], dn
            ),
            "top100_overlap": _spearman([r["top100_overlap"] for r in rs], dn),
        }
        d["H4_dense_vs_hybrid_delta_ndcg"] = {
            cond: {
                "pearson": float(
                    np.corrcoef(dn, [r[f"delta_ndcg@10_{cond}"] for r in rs])[0, 1]
                ),
                "spearman": _spearman(dn, [r[f"delta_ndcg@10_{cond}"] for r in rs])[
                    "rho"
                ],
                "dense_losses_also_lost": sum(
                    r["delta_ndcg@10_D"] < -1e-12
                    and r[f"delta_ndcg@10_{cond}"] < -1e-12
                    for r in rs
                ),
                "dense_losses": sum(r["delta_ndcg@10_D"] < -1e-12 for r in rs),
            }
            for cond in ("A", "B")
        }
        groups = {}
        for key in ("char_length", "lexical_overlap"):
            t = _tertiles(rs, key)
            groups[key] = {
                "cuts": t["cuts"],
                **{
                    g: _bootstrap_mean([r["delta_ndcg@10_D"] for r in v])
                    for g, v in t["groups"].items()
                },
            }
        groups["relevant_count"] = {
            "1": _bootstrap_mean(
                [r["delta_ndcg@10_D"] for r in rs if r["relevant_count"] == 1]
            ),
            "2+": _bootstrap_mean(
                [r["delta_ndcg@10_D"] for r in rs if r["relevant_count"] >= 2]
            ),
        }
        groups["finance_term"] = {
            "true": _bootstrap_mean(
                [r["delta_ndcg@10_D"] for r in rs if r["finance_term"]]
            ),
            "false": _bootstrap_mean(
                [r["delta_ndcg@10_D"] for r in rs if not r["finance_term"]]
            ),
        }
        d["dense_delta_ndcg_by_query_feature"] = groups
        out["datasets"][dataset] = d
    review = read_json(evidence_dir / "manual-review.json")
    tally = {}
    for case in review["cases"]:
        for label in case["labels"]:
            tally.setdefault(case["group"], {}).setdefault(label, 0)
            tally[case["group"]][label] += 1
    out["manual_review_tally"] = tally
    out["posthoc_same_source"] = read_json(evidence_dir / "posthoc-same-source.json")
    rows_by_dataset = {ds: [r for r in rows if r["dataset"] == ds] for ds in DATASETS}
    out["posthoc_v2_review"] = {
        "kind": "post-hoc, added in the PR #43 follow-up review; not preregistered",
        "score_scale_vs_separation": {
            ds: score_scale_vs_separation(rs) for ds, rs in rows_by_dataset.items()
        },
        "lexical_overlap_outlier_check": {
            ds: overlap_outlier_check(rs) for ds, rs in rows_by_dataset.items()
        },
        "ranking_parity_summary": {
            ds: v["encoders"]
            for ds, v in read_json(evidence_dir / "ranking-parity.json")[
                "datasets"
            ].items()
        },
        "v1_numpy_ties_vs_v2_canonical": read_json(
            evidence_dir / "v1-v2-comparison.json"
        )["comparison"],
    }
    return out


# ------------------------------------------------------------------- figures
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {"bccard": "#2a78d6", "strategyqa": "#eb6834"}
NAMES = {"bccard": "BC Card Test (721)", "strategyqa": "Ko-StrategyQA (592)"}
FONT = (
    "-apple-system, 'Apple SD Gothic Neo', 'Malgun Gothic', 'Noto Sans KR', "
    "'Segoe UI', Helvetica, Arial, sans-serif"
)


def _svg(width, height, body, title):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{title}" font-family="{FONT}">\n'
        f"<title>{title}</title>\n"
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def movement_figure(analysis):
    labels = (
        ("entered_top10", "top10 진입"),
        ("up_in_top10", "top10 안 상승"),
        ("same_in_top10", "top10 안 동일"),
        ("down_in_top10", "top10 안 하락"),
        ("left_top10", "top10 이탈"),
        ("outside_both", "둘 다 top10 밖"),
    )
    width, top, row_h, x0, x1 = 760, 84, 40, 190, 700
    height = top + row_h * len(labels) + 40
    shares = {
        ds: {
            k: v / analysis["datasets"][ds]["queries"]
            for k, v in analysis["datasets"][ds]["dense_rank_movement"].items()
        }
        for ds in DATASETS
    }
    hi = max(max(s.values()) for s in shares.values())
    sx = lambda v: x0 + v / hi * (x1 - x0)  # noqa: E731
    body = [
        f'<text x="20" y="26" font-size="15" font-weight="600" fill="{INK}">Dense-only 첫 relevant 문서 순위 이동 (원본 BGE → Finance)</text>',
        f'<text x="20" y="44" font-size="12" fill="{MUTED}">데이터셋별 query 비율. 막대 끝 숫자는 query 수.</text>',
    ]
    for i, ds in enumerate(DATASETS):
        body.append(
            f'<rect x="{20 + i * 220}" y="56" width="14" height="10" rx="2" fill="{SERIES[ds]}"/><text x="{40 + i * 220}" y="65" font-size="11" fill="{INK}">{NAMES[ds]}</text>'
        )
    for j, (key, label) in enumerate(labels):
        y = top + j * row_h
        body.append(
            f'<text x="{x0 - 10}" y="{y + 14}" font-size="12" text-anchor="end" fill="{INK}">{label}</text>'
        )
        for i, ds in enumerate(DATASETS):
            share = shares[ds][key]
            count = analysis["datasets"][ds]["dense_rank_movement"][key]
            yy = y + 2 + i * 14
            body.append(
                f'<rect x="{x0}" y="{yy}" width="{max(sx(share) - x0, 1):.1f}" height="11" rx="2" fill="{SERIES[ds]}"><title>{NAMES[ds]} {label}: {count} ({share:.1%})</title></rect>'
            )
            body.append(
                f'<text x="{sx(share) + 4:.1f}" y="{yy + 9}" font-size="10" fill="{MUTED}">{count}</text>'
            )
    return _svg(width, height, body, "Dense first relevant rank movement")


def drift_figure(rows):
    """Mean Dense delta nDCG by drift quintile per dataset."""
    import numpy as np

    width, height, x0, x1, y0, y1 = 760, 380, 80, 720, 80, 320
    body = [
        f'<text x="20" y="26" font-size="15" font-weight="600" fill="{INK}">Query embedding drift와 Dense ΔnDCG@10</text>',
        f'<text x="20" y="44" font-size="12" fill="{MUTED}">데이터셋별 drift(1 − cos) 5분위. 점 = 구간 평균 ΔnDCG@10, x = 구간 drift 중앙값. 탐색적 연관임.</text>',
    ]
    points = {}
    for ds in DATASETS:
        rs = [r for r in rows if r["dataset"] == ds]
        drift = np.asarray([1 - r["query_cosine_bge_fin"] for r in rs])
        delta = np.asarray([r["delta_ndcg@10_D"] for r in rs])
        edges = np.quantile(drift, np.linspace(0, 1, 6))
        bins = np.clip(np.searchsorted(edges, drift, side="right") - 1, 0, 4)
        points[ds] = [
            (
                float(np.median(drift[bins == b])),
                float(delta[bins == b].mean()),
                int((bins == b).sum()),
            )
            for b in range(5)
        ]
    xs = [p[0] for v in points.values() for p in v]
    ys = [p[1] for v in points.values() for p in v] + [0.0]
    xlo, xhi = 0.0, max(xs) * 1.08
    ylo, yhi = min(ys) - 0.01, max(ys) + 0.01
    sx = lambda v: x0 + (v - xlo) / (xhi - xlo) * (x1 - x0)  # noqa: E731
    sy = lambda v: y1 - (v - ylo) / (yhi - ylo) * (y1 - y0)  # noqa: E731
    body.append(
        f'<line x1="{x0}" x2="{x1}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" stroke="{MUTED}" stroke-width="1"/>'
    )
    for t in np.linspace(ylo, yhi, 5):
        body.append(
            f'<text x="{x0 - 8}" y="{sy(t) + 4:.1f}" font-size="11" text-anchor="end" fill="{MUTED}">{t:+.3f}</text>'
        )
    for t in np.linspace(xlo, xhi, 5):
        body.append(
            f'<text x="{sx(t):.1f}" y="{y1 + 18}" font-size="11" text-anchor="middle" fill="{MUTED}">{t:.4f}</text>'
        )
    body.append(
        f'<text x="{(x0 + x1) / 2}" y="{y1 + 36}" font-size="11" text-anchor="middle" fill="{MUTED}">query drift 1 − cos(q_bge, q_fin)</text>'
    )
    for i, (ds, pts) in enumerate(points.items()):
        path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _ in pts)
        body.append(
            f'<polyline points="{path}" fill="none" stroke="{SERIES[ds]}" stroke-width="2"/>'
        )
        for x, y, n in pts:
            body.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{SERIES[ds]}" stroke="{SURFACE}" stroke-width="2"><title>{NAMES[ds]} drift {x:.4f}: ΔnDCG {y:+.4f} (n={n})</title></circle>'
            )
        lx, ly, _ = pts[-1]
        body.append(
            f'<text x="{sx(lx) - 8:.1f}" y="{sy(ly) - 10:.1f}" font-size="11" text-anchor="end" fill="{INK}">{NAMES[ds]}</text>'
        )
    return _svg(width, height, body, "Drift versus Dense delta nDCG")


def write_manifest(evidence_dir=EVIDENCE):
    lines = [
        f"{sha256(p)}  {p.name}"
        for p in sorted(Path(evidence_dir).iterdir())
        if p.is_file() and p.name != "MANIFEST.sha256"
    ]
    (Path(evidence_dir) / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def write_analysis(evidence_dir=EVIDENCE):
    evidence_dir = Path(evidence_dir)
    result = analyze(evidence_dir)
    (evidence_dir / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    rows = read_rows(evidence_dir / "per-query-summary.jsonl")
    (evidence_dir / "rank-movement.svg").write_text(movement_figure(result))
    (evidence_dir / "drift-vs-delta.svg").write_text(drift_figure(rows))
    write_manifest(evidence_dir)
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
    evidence_dir = Path(evidence_dir)
    listed = {}
    for line in (evidence_dir / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        listed[name] = digest
        if sha256(evidence_dir / name) != digest:
            raise ValueError(f"manifest mismatch: {name}")
    present = {p.name for p in evidence_dir.iterdir() if p.is_file()} - {
        "MANIFEST.sha256"
    }
    if set(listed) != present:
        raise ValueError(f"unlisted or missing files: {sorted(set(listed) ^ present)}")
    # NumPy/SciPy reductions can differ in the last bits across CPUs.
    if not _close(
        json.loads(json.dumps(analyze(evidence_dir))),
        read_json(evidence_dir / "analysis.json"),
    ):
        raise ValueError("analysis.json does not match recomputation")
    recomputed = json.loads(json.dumps(analyze(evidence_dir)))
    if (evidence_dir / "rank-movement.svg").read_text() != movement_figure(recomputed):
        raise ValueError("rank movement figure differs")
    rows = read_rows(evidence_dir / "per-query-summary.jsonl")
    if (evidence_dir / "drift-vs-delta.svg").read_text() != drift_figure(rows):
        raise ValueError("drift figure differs")
    return {"verified": True, "files": len(listed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        choices=(
            "features",
            "cases",
            "posthoc",
            "ranking-parity",
            "v1-records",
            "analyze",
            "check",
        ),
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--external-cache", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "features":
        result = compute_features(args)
    elif args.command == "cases":
        print_cases(args)
        return 0
    elif args.command == "posthoc":
        result = posthoc_same_source(args)
    elif args.command == "ranking-parity":
        result = ranking_parity(args)
    elif args.command == "v1-records":
        result = write_v1_records()
    elif args.command == "analyze":
        result = {"datasets": list(write_analysis()["datasets"])}
    else:
        result = check()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

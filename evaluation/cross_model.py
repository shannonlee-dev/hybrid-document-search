"""Join per-query rankings from several runs and count where encoders disagree.

Runs are `{query_id: row}` maps with the enterprise report row shape
(`results`, `metrics`, optionally `mode`). Every count is descriptive and is
computed on the query IDs passed in; nothing here selects a method.
"""

from collections import Counter
from math import fsum

TOP_K = 10
DELTA_BINS = (
    "-[0.5,1]",
    "-[0.2,0.5)",
    "-(0,0.2)",
    "tie",
    "+(0,0.2)",
    "+[0.2,0.5)",
    "+[0.5,1]",
)
_TIE = 1e-12


def _ids(row, k=TOP_K):
    return [hit["document_id"] for hit in row["results"][:k]]


def relevant_rank(row, judgments, k=TOP_K):
    """1-based rank of the first relevant document in the top k, or None."""
    for rank, doc in enumerate(_ids(row, k), start=1):
        if judgments.get(doc, 0) > 0:
            return rank
    return None


def query_table(runs, qrels, texts, query_ids, *, pairs):
    """One compact row per query: relevant rank and nDCG@10 per run, plus overlaps."""
    rows = []
    for q in query_ids:
        judged = qrels[q]
        row = {
            "query_id": q,
            "text": texts[q],
            "relevant": sorted(d for d, rel in judged.items() if rel > 0),
        }
        for name, run in runs.items():
            row[f"{name}_rank"] = relevant_rank(run[q], judged)
            row[f"{name}_ndcg@10"] = run[q]["metrics"]["ndcg@10"]
            if "mode" in run[q]:
                row[f"{name}_mode"] = run[q]["mode"]
        for left, right in pairs:
            a, b = _ids(runs[left][q]), _ids(runs[right][q])
            row[f"top1_differs:{left}|{right}"] = a[:1] != b[:1]
            row[f"top10_overlap:{left}|{right}"] = len(set(a) & set(b))
        rows.append(row)
    return rows


def _hit(row, name):
    return row[f"{name}_rank"] is not None


def _delta_bin(delta):
    if abs(delta) <= _TIE:
        return "tie"
    size = abs(delta)
    bucket = "(0,0.2)" if size < 0.2 else "[0.2,0.5)" if size < 0.5 else "[0.5,1]"
    return ("+" if delta > 0 else "-") + bucket


def _delta_histogram(deltas):
    counts = Counter(_delta_bin(value) for value in deltas)
    return {name: counts[name] for name in DELTA_BINS}


def encoder_agreement(rows, first, second):
    """2x2 counts of two dense runs on hit@10 and on P@1."""
    result = {}
    for label, test in (
        ("hit@10", lambda r, n: _hit(r, n)),
        ("p@1", lambda r, n: r[f"{n}_rank"] == 1),
    ):
        result[label] = {
            "both": sum(test(r, first) and test(r, second) for r in rows),
            f"{first}_only": sum(test(r, first) and not test(r, second) for r in rows),
            f"{second}_only": sum(test(r, second) and not test(r, first) for r in rows),
            "neither": sum(not test(r, first) and not test(r, second) for r in rows),
        }
    return result


def sparse_complementarity(rows, sparse, dense_runs):
    """Queries where BM25 finds the answer and a dense run does not."""
    only = {
        dense: {r["query_id"] for r in rows if _hit(r, sparse) and not _hit(r, dense)}
        for dense in dense_runs
    }
    top1 = {
        dense: {
            r["query_id"]
            for r in rows
            if r[f"{sparse}_rank"] == 1 and r[f"{dense}_rank"] != 1
        }
        for dense in dense_runs
    }
    shared = set.intersection(*only.values()) if only else set()
    shared_top1 = set.intersection(*top1.values()) if top1 else set()
    return {
        "bm25_hit@10_dense_miss": {name: len(ids) for name, ids in only.items()},
        "bm25_hit@10_all_dense_miss": len(shared),
        "bm25_top1_dense_not_top1": {name: len(ids) for name, ids in top1.items()},
        "bm25_top1_all_dense_not_top1": len(shared_top1),
        "query_ids": {
            "bm25_hit@10_all_dense_miss": sorted(shared),
            "bm25_top1_all_dense_not_top1": sorted(shared_top1),
        },
    }


def fusion_effect(rows, sparse, dense, fused):
    """Improved / regressed / tied queries when the fixed sparse weight is added."""
    deltas = {
        r["query_id"]: r[f"{fused}_ndcg@10"] - r[f"{dense}_ndcg@10"] for r in rows
    }
    by_id = {r["query_id"]: r for r in rows}
    improved = [q for q, d in deltas.items() if d > _TIE]
    regressed = [q for q, d in deltas.items() if d < -_TIE]

    def bm25_rank_profile(ids):
        ranks = [by_id[q][f"{sparse}_rank"] for q in ids]
        return {
            "bm25_rank_1": sum(rank == 1 for rank in ranks),
            "bm25_rank_2_10": sum(rank is not None and rank > 1 for rank in ranks),
            "bm25_miss": sum(rank is None for rank in ranks),
        }

    return {
        "improved": len(improved),
        "regressed": len(regressed),
        "tied": len(rows) - len(improved) - len(regressed),
        "gain_sum": fsum(deltas[q] for q in improved),
        "loss_sum": fsum(deltas[q] for q in regressed),
        "delta_histogram": _delta_histogram(deltas.values()),
        "improved_bm25_rank": bm25_rank_profile(improved),
        "regressed_bm25_rank": bm25_rank_profile(regressed),
        "rescued_from_dense_miss": sum(
            not _hit(by_id[q], dense) and _hit(by_id[q], fused) for q in deltas
        ),
        "lost_from_dense_hit": sum(
            _hit(by_id[q], dense) and not _hit(by_id[q], fused) for q in deltas
        ),
    }


def method_oracle(rows, methods, reference):
    """Post-hoc per-query best of `methods`; ties keep `reference`."""
    order = (reference, *[m for m in methods if m != reference])
    choice = {
        r["query_id"]: max(order, key=lambda m, r=r: r[f"{m}_ndcg@10"]) for r in rows
    }
    by_id = {r["query_id"]: r for r in rows}
    gains = {
        q: by_id[q][f"{m}_ndcg@10"] - by_id[q][f"{reference}_ndcg@10"]
        for q, m in choice.items()
    }
    count = len(rows)
    return {
        "reference": reference,
        "mean_reference": fsum(r[f"{reference}_ndcg@10"] for r in rows) / count,
        "mean_oracle": fsum(by_id[q][f"{m}_ndcg@10"] for q, m in choice.items())
        / count,
        "choice_counts": dict(Counter(choice.values())),
        "headroom_queries": sorted(q for q, g in gains.items() if g > _TIE),
    }


def adaptive_recovery(rows, *, fixed, adaptive, oracle, disagreement):
    """How much of the mode-oracle headroom over Fixed the router actually took."""
    by_id = {r["query_id"]: r for r in rows}
    headroom = [
        q
        for q, r in by_id.items()
        if r[f"{oracle}_ndcg@10"] > r[f"{fixed}_ndcg@10"] + _TIE
    ]
    recovered = [
        q
        for q in headroom
        if by_id[q][f"{adaptive}_ndcg@10"] > by_id[q][f"{fixed}_ndcg@10"] + _TIE
    ]
    agree = sum(r[f"{adaptive}_mode"] == r[f"{oracle}_mode"] for r in rows)

    def gain(name, ids):
        if not ids:
            return None
        return fsum(
            by_id[q][f"{name}_ndcg@10"] - by_id[q][f"{fixed}_ndcg@10"] for q in ids
        ) / len(ids)

    groups = {
        "top1_differs": [q for q, r in by_id.items() if r[disagreement]],
        "top1_same": [q for q, r in by_id.items() if not r[disagreement]],
    }
    return {
        "headroom_queries": len(headroom),
        "router_improved_on_headroom": len(recovered),
        "router_mode_counts": dict(Counter(r[f"{adaptive}_mode"] for r in rows)),
        "oracle_mode_counts": dict(Counter(r[f"{oracle}_mode"] for r in rows)),
        "mode_agreement": agree / len(rows),
        "by_bm25_dense_top1": {
            name: {
                "query_count": len(ids),
                "headroom_queries": len(set(ids) & set(headroom)),
                "adaptive_minus_fixed": gain(adaptive, ids),
                "oracle_minus_fixed": gain(oracle, ids),
            }
            for name, ids in groups.items()
        },
    }

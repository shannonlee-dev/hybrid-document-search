"""External baseline review: Finance vs Kanana on committed per-query results.

Protocol: docs/validation/2026-11-04/external-baseline/protocol.md (fixed before
computing). `leakage` needs the prepared BC Card data and scikit-learn;
`analyze` and `check` read committed files only.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

from evaluation.paired import paired_comparison

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/validation/2026-11-04/external-baseline"
PHASE5_CSV = ROOT / "docs/experiments/phase5/primary-queries.csv"
SERVICE_ROWS = (
    ROOT / "docs/validation/2026-11-02/finance-service-eval/bccard-per-query.jsonl"
)
TIE_ROWS = (
    ROOT
    / "docs/validation/2026-11-02/finance-strategyqa-analysis/tie-sensitivity-per-query.jsonl"
)
METRICS = ("ndcg@10", "recall@5", "recall@10", "mrr@10", "p@1")
SEED, RESAMPLES = 20261102, 20_000
NEAR_DUP = 0.90
BANDS = (0.80, 0.90, 0.95)

# (name, A run, B run); difference is B - A.
PRIMARY = (
    ("P1_dense", "kanana_dense", "fin_dense"),
    ("P2_fixed05", "kanana_fixed", "fin_fixed"),
)
REFERENCE = (
    ("R_bge_kanana_dense", "bge_dense", "kanana_dense"),
    ("R_bge_kanana_fixed", "bge_fixed", "kanana_fixed"),
    ("R_bge_fin_dense", "bge_dense", "fin_dense"),
    ("R_bge_fin_fixed", "bge_fixed", "fin_fixed"),
)


def metrics_from_rank(rank):
    """Metrics for a query with exactly one relevant document at `rank` (or None)."""
    if rank is None:
        return dict.fromkeys(METRICS, 0.0)
    return {
        "ndcg@10": 1 / math.log2(rank + 1),
        "recall@5": float(rank <= 5),
        "recall@10": 1.0,
        "mrr@10": 1 / rank,
        "p@1": float(rank == 1),
    }


def _rank(value):
    return None if value in ("", None) else int(value)


def load_runs():
    """Return (query_ids, {run: {query_id: metrics}}, gate) from committed rows."""
    rows = list(csv.DictReader(PHASE5_CSV.open(encoding="utf-8")))
    ids = [r["query_id"] for r in rows]
    service = {}
    for line in SERVICE_ROWS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        service.setdefault(r["condition"], {})[r["query_id"]] = r
    runs, ranks, derived_mismatch = {}, {}, {}
    for name in ("bge_dense", "bge_fixed", "kanana_dense", "kanana_fixed", "bm25"):
        ranks[name] = {r["query_id"]: _rank(r[f"{name}_rank"]) for r in rows}
        runs[name] = {q: metrics_from_rank(k) for q, k in ranks[name].items()}
        derived_mismatch[name] = sum(
            abs(runs[name][r["query_id"]]["ndcg@10"] - float(r[f"{name}_ndcg@10"]))
            > 1e-12
            for r in rows
        )
    for name, cond in (
        ("fin_dense", "D-fin"),
        ("fin_fixed", "A-fin"),
        ("svc_bge_dense", "D-bge"),
        ("svc_bge_fixed", "A-bge"),
    ):
        ranks[name] = {q: service[cond][q]["relevant_rank_top10"] for q in ids}
        runs[name] = {q: metrics_from_rank(k) for q, k in ranks[name].items()}
        derived_mismatch[name] = sum(
            abs(runs[name][q][m] - service[cond][q][m]) > 1e-12
            for q in ids
            for m in METRICS
        )
    gate = {
        "query_count": len(ids),
        "service_query_order_equal": all(list(service[c]) == ids for c in service),
        "single_relevant_per_query": all(len(r["relevant"].split()) == 1 for r in rows),
        "derived_metric_mismatches": derived_mismatch,
        "phase5_vs_service_rank_mismatches": {
            "bge_dense~D-bge": sum(
                ranks["bge_dense"][q] != ranks["svc_bge_dense"][q] for q in ids
            ),
            "bge_fixed~A-bge": sum(
                ranks["bge_fixed"][q] != ranks["svc_bge_fixed"][q] for q in ids
            ),
        },
    }
    gate["passed"] = (
        gate["service_query_order_equal"]
        and gate["single_relevant_per_query"]
        and not any(derived_mismatch.values())
        and not any(gate["phase5_vs_service_rank_mismatches"].values())
    )
    return ids, runs, gate


def concentration(a, b, ids, top=10):
    delta = sorted(((b[q]["ndcg@10"] - a[q]["ndcg@10"], q) for q in ids), reverse=True)
    gains = [d for d, _ in delta if d > 1e-12]
    losses = [d for d, _ in delta if d < -1e-12]
    by_abs = sorted(delta, key=lambda x: -abs(x[0]))
    rest = [d for d, _ in by_abs[top:]]
    return {
        "nonzero_queries": len(gains) + len(losses),
        "gain_sum": math.fsum(gains),
        "loss_sum": math.fsum(losses),
        f"top{top}_gain_share_of_gain_sum": math.fsum(gains[:top]) / math.fsum(gains)
        if gains
        else None,
        f"mean_without_top{top}_abs": math.fsum(rest) / len(rest),
    }


def _compare(runs, a, b, ids, seed=SEED, resamples=RESAMPLES):
    return paired_comparison(
        runs[a], runs[b], ids, metrics=METRICS, seed=seed, resamples=resamples
    )


def _bonferroni(runs, a, b, ids):
    import numpy as np

    delta = np.array([runs[b][q]["ndcg@10"] - runs[a][q]["ndcg@10"] for q in ids])
    idx = np.random.default_rng(SEED).integers(0, len(ids), size=(RESAMPLES, len(ids)))
    low, high = np.percentile(delta[idx].mean(axis=1), [1.25, 98.75])
    return [float(low), float(high)]


def _verdict(ci):
    if ci[0] > 0:
        return "B(Finance) 우위"
    if ci[1] < 0:
        return "A(Kanana) 우위"
    return "차이 확인 안 됨"


def label_conflict_queries():
    out = set()
    for line in TIE_ROWS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["binary_label_conflict_query"]:
            out.add(r["query_id"])
    return sorted(out)


def analyze():
    ids, runs, gate = load_runs()
    if not gate["passed"]:
        return {"gate": gate, "stopped": "연결 조건 불일치"}
    result = {
        "protocol": "docs/validation/2026-11-04/external-baseline/protocol.md",
        "gate": gate,
        "bootstrap": {"seed": SEED, "resamples": RESAMPLES},
        "means": {
            name: {
                m: math.fsum(runs[name][q][m] for q in ids) / len(ids) for m in METRICS
            }
            for name in (
                "bm25",
                "bge_dense",
                "bge_fixed",
                "kanana_dense",
                "kanana_fixed",
                "fin_dense",
                "fin_fixed",
            )
        },
        "primary": {},
        "reference": {},
        "sensitivity": {},
    }
    for name, a, b in PRIMARY:
        cmp = _compare(runs, a, b, ids)
        cmp["a"], cmp["b"] = a, b
        cmp["ndcg_bonferroni_97_5"] = _bonferroni(runs, a, b, ids)
        cmp["verdict_ndcg"] = _verdict(cmp["metrics"]["ndcg@10"]["ci95"])
        cmp["concentration"] = concentration(runs[a], runs[b], ids)
        result["primary"][name] = cmp
    for name, a, b in REFERENCE:
        cmp = _compare(runs, a, b, ids)
        cmp["a"], cmp["b"] = a, b
        cmp["concentration"] = concentration(runs[a], runs[b], ids)
        result["reference"][name] = cmp
    # Same seeds as the source documents, to confirm the implementation.
    result["reference_replay"] = {
        "phase5_bge_kanana_dense_seed20261030_10000": _compare(
            runs, "bge_dense", "kanana_dense", ids, seed=20261030, resamples=10_000
        )["metrics"]["ndcg@10"],
        "pr42_bge_fin_fixed_seed20261102_20000": _compare(
            runs, "bge_fixed", "fin_fixed", ids
        )["metrics"]["ndcg@10"],
    }
    conflicts = label_conflict_queries()
    kept = [q for q in ids if q not in set(conflicts)]
    result["sensitivity"]["exclude_label_conflict"] = {
        "excluded": conflicts,
        "query_count": len(kept),
        "comparisons": {
            name: _compare(runs, a, b, kept)["metrics"]["ndcg@10"]
            for name, a, b in PRIMARY
        },
    }
    leakage_path = EVIDENCE / "leakage.json"
    if leakage_path.exists():
        leak = json.loads(leakage_path.read_text(encoding="utf-8"))
        near = set(leak["query_side"]["near_duplicate_query_ids"])
        kept = [q for q in ids if q not in near]
        result["sensitivity"]["exclude_near_duplicate_train_query"] = {
            "threshold": leak["threshold"],
            "excluded_count": len(ids) - len(kept),
            "query_count": len(kept),
            "comparisons": {
                name: _compare(runs, a, b, kept)["metrics"]["ndcg@10"]
                for name, a, b in PRIMARY + REFERENCE[2:]
            },
        }
    return result


_SPACE = re.compile(r"\s+")


def _norm(text):
    return _SPACE.sub(" ", text.casefold()).strip()


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _max_similarity(left, right):
    """Max char 3-gram TF-IDF cosine of each `left` text against `right`."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3)).fit(right)
    lm, rm = vec.transform(left), vec.transform(right).T.tocsc()
    best = np.zeros(len(left))
    for start in range(0, len(left), 256):
        block = (lm[start : start + 256] @ rm).toarray()
        best[start : start + 256] = block.max(axis=1)
    return best


def leakage(data_dir):
    data = Path(data_dir)
    ids, _, _ = load_runs()
    test = {r["query_id"]: r for r in _jsonl(data / "queries_test.jsonl")}
    train = _jsonl(data / "queries_train.jsonl")
    corpus = {r["document_id"]: r["text"] for r in _jsonl(data / "corpus.jsonl")}
    gold = {
        r["query_id"]: r["document_id"]
        for r in _jsonl(data / "qrels_test.jsonl")
        if r["relevance"] > 0 and r["query_id"] in set(ids)
    }
    train_pos = sorted(
        {
            r["document_id"]
            for r in _jsonl(data / "qrels_train.jsonl")
            if r["relevance"] == 3
        }
        - set(gold.values())
    )
    q_sim = _max_similarity(
        [_norm(test[q]["text"]) for q in ids], [_norm(r["text"]) for r in train]
    )
    d_sim = _max_similarity(
        [_norm(corpus[gold[q]]) for q in ids], [_norm(corpus[d]) for d in train_pos]
    )

    def bands(sims):
        return {f">={t:.2f}": int((sims >= t).sum()) for t in BANDS}

    exact = {_norm(r["text"]) for r in train}
    return {
        "protocol": "docs/validation/2026-11-04/external-baseline/protocol.md",
        "method": "casefold+whitespace, char_wb 3-gram TF-IDF cosine fit on train side",
        "threshold": NEAR_DUP,
        "query_side": {
            "test_queries": len(ids),
            "train_queries": len(train),
            "exact_normalized_match": sum(_norm(test[q]["text"]) in exact for q in ids),
            "text_in_train_flag": sum(bool(test[q].get("text_in_train")) for q in ids),
            "bands": bands(q_sim),
            "near_duplicate_query_ids": [
                q for q, s in zip(ids, q_sim) if s >= NEAR_DUP
            ],
            "max_similarity_by_query": {
                q: round(float(s), 6) for q, s in zip(ids, q_sim)
            },
        },
        "document_side": {
            "test_gold_chunks": len(gold),
            "train_positive_chunks_excluding_test_gold": len(train_pos),
            "bands": bands(d_sim),
        },
    }


def _write(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def check():
    committed = json.loads((EVIDENCE / "comparison.json").read_text(encoding="utf-8"))
    fresh = json.loads(json.dumps(analyze(), ensure_ascii=False))
    return {"verified": committed == fresh}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    leak = sub.add_parser("leakage")
    leak.add_argument("--data-dir", default="data/processed/bccard")
    sub.add_parser("analyze")
    sub.add_parser("check")
    args = parser.parse_args(argv)
    if args.command == "leakage":
        _write(EVIDENCE / "leakage.json", leakage(args.data_dir))
    elif args.command == "analyze":
        _write(EVIDENCE / "comparison.json", analyze())
    else:
        result = check()
        print(result)
        if not result["verified"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

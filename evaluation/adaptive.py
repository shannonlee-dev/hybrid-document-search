"""Issue #24: finite train/validation selection, frozen test and failure evidence."""

from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from data.bccard import load_query_metadata
from evaluation.benchmark import _code_provenance, _environment, benchmark_retriever
from evaluation.data import file_sha256, load_evaluation_dataset
from evaluation.enterprise import (
    _build_retrievers,
    _load_vectors,
    query_metrics,
    run_enterprise_baselines,
    slice_definitions,
    summarize,
)
from evaluation.json_io import read_json, write_json
from evaluation.latency import latency_summary
from evaluation.metrics import evaluate_query, ndcg_at_k
from fusion.adaptive import (
    MODES,
    AdaptiveRetriever,
    Router,
    query_features,
    training_labels,
)
from fusion.score import _validate_weight

DEFAULT_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "config/adaptive_candidates.json"
)
FIXED_WEIGHT = 0.05
KINDS = ("logistic", "mlp", "tree")
CODE_FILES = (
    "fusion/adaptive.py",
    "evaluation/adaptive.py",
    "scripts/evaluate_adaptive.py",
    "fusion/score.py",
    "evaluation/enterprise.py",
    "evaluation/data.py",
    "evaluation/metrics.py",
    "evaluation/benchmark.py",
    "evaluation/latency.py",
    "retrievers/bm25.py",
    "retrievers/dense.py",
    "data/bccard.py",
    "indexing/vectors.py",
    "scripts/benchmark_score_fusion.py",
    "evaluation/json_io.py",
    "evaluation/runtime_config.py",
)


def _code_hashes():
    root = Path(__file__).resolve().parents[1]
    return _code_provenance()["sha256"] | {
        name: file_sha256(root / name) for name in CODE_FILES
    }


def _retriever(data_dir, cache_dir, split, router):
    dataset = load_evaluation_dataset(data_dir, split)
    vectors, queries, record = _load_vectors(data_dir, cache_dir, dataset)
    sources = _build_retrievers(dataset, vectors, queries)
    return (
        dataset,
        AdaptiveRetriever(sources["bm25"][0], sources["dense"][0], router),
        record,
    )


def _constant_router():
    return Router.fit(
        np.zeros((1, 10)),
        np.ones((1, 3)),
        weights=(0.2, FIXED_WEIGHT, 0.0),
        kind="logistic",
        split="train",
    )


def _load_weight_sets(path):
    config = read_json(path)
    weight_sets = config.get("weight_sets") if isinstance(config, dict) else None
    if not isinstance(weight_sets, list) or not weight_sets:
        raise ValueError("weight_sets must be a non-empty list")
    for triple in weight_sets:
        if not isinstance(triple, list) or len(triple) != 3:
            raise ValueError("each weight set requires three weights")
        for weight in triple:
            _validate_weight(weight)
        if not triple[0] > triple[1] > triple[2] >= 0:
            raise ValueError("weights must descend and be non-negative")
    return weight_sets


def collect_training_table(
    data_dir, cache_dir, split, progress=None, *, weight_sets=None, table_path=None
):
    if split not in ("train", "validation"):
        raise ValueError("candidate collection accepts train or validation only")
    dataset, retriever, _ = _retriever(data_dir, cache_dir, split, _constant_router())
    if weight_sets is None:
        weight_sets = _load_weight_sets(DEFAULT_CANDIDATES)
    weights = sorted({FIXED_WEIGHT, *(w for triple in weight_sets for w in triple)})
    identity = {
        "split": split,
        "weights": weights,
        "dataset": dataset.provenance,
        "encoding_sha256": file_sha256(Path(cache_dir) / "encoding.json"),
        "code_sha256": _code_hashes(),
    }
    features, quality = [], []
    if table_path is not None and Path(table_path).exists():
        cached = read_json(table_path)
        if cached["identity"] != identity:
            raise ValueError("training table cache does not match current inputs")
        features, quality = cached["features"], cached["quality"]
        if len(features) != len(quality) or len(features) > len(dataset.queries):
            raise ValueError("training table cache row counts do not match")
        if progress:
            progress(
                f"{split}: resumed {len(features)}/{len(dataset.queries)} cached rows"
            )

    def checkpoint():
        if table_path is not None:
            write_json(
                Path(table_path),
                {"identity": identity, "features": features, "quality": quality},
            )

    start = len(features)
    for index, query in enumerate(dataset.queries[start:], start):
        sparse, dense = retriever.score_arrays(query.text)
        features.append(query_features(query.text, sparse, dense, retriever._ids))
        quality.append(
            [
                ndcg_at_k(
                    retriever.rank_scores(sparse, dense, weight, 10),
                    dataset.qrels[query.query_id],
                    10,
                )
                for weight in weights
            ]
        )
        if (index + 1) % 500 == 0:
            checkpoint()
            if progress:
                progress(f"{split}: {index + 1}/{len(dataset.queries)}")
    checkpoint()
    return dataset, np.asarray(features), np.asarray(quality), weights


def fit_router(
    data_dir,
    cache_dir,
    output_dir,
    progress=None,
    *,
    candidates_path=DEFAULT_CANDIDATES,
    table_dir=None,
):
    """Use every official train row; choose only by non-synthetic validation nDCG."""
    output = Path(output_dir)
    if output.exists():
        raise ValueError("run directory already exists; preserve prior evidence")
    weight_sets = _load_weight_sets(candidates_path)
    output.mkdir(parents=True)
    table_dir = Path(table_dir) if table_dir is not None else output
    protocol = {
        "weight_sets": weight_sets,
        "fixed_baseline_weight": FIXED_WEIGHT,
        "reporting": "retain all outcomes including no improvement; no test-based retuning",
        "classifiers": KINDS,
        "seed": 42,
        "objective": "binary nDCG@10",
        "label_ties": "balanced, keyword-heavy, semantic-heavy; atol=1e-12",
        "selection": "non_synthetic validation mean nDCG@10; ties keep first candidate",
        "candidate_order": "weight sets in listed order; logistic, mlp, tree",
        "mlp": {
            "hidden_layer_sizes": [16],
            "activation": "relu",
            "solver": "lbfgs",
            "alpha": 0.0001,
            "max_iter": 500,
        },
        "train_sampling": "all official train queries in manifest order",
        "code_sha256": _code_hashes(),
        "dataset_manifest_sha256": file_sha256(Path(data_dir) / "manifest.json"),
        "encoding_sha256": file_sha256(Path(cache_dir) / "encoding.json"),
        "test_exposure": "official test was inspected in prior baseline/exploratory work; not fresh independent confirmation",
    }
    write_json(output / "protocol.json", protocol)
    train, x, tq, weights = collect_training_table(
        data_dir,
        cache_dir,
        "train",
        progress,
        weight_sets=weight_sets,
        table_path=table_dir / "train-table.json",
    )
    validation, vx, vq, _ = collect_training_table(
        data_dir,
        cache_dir,
        "validation",
        progress,
        weight_sets=weight_sets,
        table_path=table_dir / "validation-table.json",
    )
    primary = set(
        slice_definitions(load_query_metadata(data_dir, "validation"))["non_synthetic"]
    )
    mask = np.asarray([q.query_id in primary for q in validation.queries])
    if not mask.any():
        raise ValueError("validation requires non-synthetic queries")
    best, candidates = None, []
    for triple in weight_sets:
        indices = [weights.index(w) for w in triple]
        labels = training_labels(tq[:, indices])
        for kind in KINDS:
            if progress:
                progress(f"fit: {triple} / {kind}")
            model = Router.fit(
                x, tq[:, indices], weights=triple, kind=kind, split=train.split
            )
            modes = model.select(vx)
            selected_quality = vq[np.arange(len(vx)), np.asarray(indices)[modes]]
            score = float(selected_quality[mask].mean())
            candidate = {
                "weights": triple,
                "classifier": kind,
                "primary_ndcg@10": score,
                "training_label_counts": dict(Counter(MODES[int(i)] for i in labels)),
                "validation_mode_counts": dict(Counter(MODES[int(i)] for i in modes)),
            }
            candidates.append(candidate)
            if best is None or score > best[0] + 1e-12:
                best = (score, model, candidate, selected_quality, labels)
    _, model, chosen, quality, labels = best
    model.save(output / "router.json")
    write_json(
        output / "train-labels.json",
        [
            {"query_id": q.query_id, "mode": MODES[int(label)]}
            for q, label in zip(train.queries, labels, strict=True)
        ],
    )
    fixed = vq[:, weights.index(FIXED_WEIGHT)]
    write_json(
        output / "validation-selection.json",
        {
            "candidates": candidates,
            "selected": chosen,
            "fixed_primary_ndcg@10": float(fixed[mask].mean()),
            "adaptive_primary_ndcg@10": float(quality[mask].mean()),
            "queries": [
                {
                    "query_id": q.query_id,
                    "fixed_ndcg@10": float(f),
                    "adaptive_ndcg@10": float(a),
                }
                for q, f, a in zip(validation.queries, fixed, quality, strict=True)
            ],
            "interpretation": "selection evidence, not an independent statistical significance result",
        },
    )
    frozen = {
        "training_split": "train",
        "selection_split": "validation",
        "quality_inputs": {
            "train": train.provenance,
            "validation": validation.provenance,
        },
        "selected": chosen,
        "candidates": candidates,
        "router_sha256": file_sha256(output / "router.json"),
        "protocol_sha256": file_sha256(output / "protocol.json"),
        "training_labels_sha256": file_sha256(output / "train-labels.json"),
        "validation_selection_sha256": file_sha256(
            output / "validation-selection.json"
        ),
        "environment": _environment(),
        "code": _code_provenance(),
        "default_service_method": "score_fusion; no service integration in #24",
    }
    write_json(output / "frozen.json", frozen)
    return frozen


class _MeasuredRouter:
    def __init__(self, retriever):
        self.retriever = retriever
        self.details = []

    def search(self, query, top_k):
        results, details = self.retriever.search_details(query, top_k)
        self.details.append(details)
        return results


def _quality_rows(rows, dataset):
    for row in rows:
        row["metrics"] |= query_metrics(row["results"], dataset.qrels[row["query_id"]])
        row["metrics"]["mrr@10"] = row["metrics"]["rr@10"]
    return {row["query_id"]: row["metrics"] for row in rows}


def evaluate_frozen(
    data_dir, cache_dir, output_dir, *, warmup=1, repeats=5, progress=None
):
    output = Path(output_dir)
    frozen = read_json(output / "frozen.json")
    protocol = read_json(output / "protocol.json")
    for name, key in (
        ("router.json", "router_sha256"),
        ("protocol.json", "protocol_sha256"),
        ("train-labels.json", "training_labels_sha256"),
        ("validation-selection.json", "validation_selection_sha256"),
    ):
        if file_sha256(output / name) != frozen[key]:
            raise ValueError(f"frozen artifact changed: {name}")
    if _code_hashes() != protocol["code_sha256"]:
        raise ValueError("implementation changed after freeze")
    if (
        file_sha256(Path(data_dir) / "manifest.json")
        != protocol["dataset_manifest_sha256"]
    ):
        raise ValueError("dataset changed after freeze")
    if file_sha256(Path(cache_dir) / "encoding.json") != protocol["encoding_sha256"]:
        raise ValueError("encoding changed after freeze")
    if any(
        isinstance(n, bool) or not isinstance(n, int) or n < 1
        for n in (warmup, repeats)
    ):
        raise ValueError("warmup and repeats must be positive integers")
    try:
        with (output / "test-started.json").open("x", encoding="utf-8") as marker:
            import json

            json.dump({"frozen_sha256": file_sha256(output / "frozen.json")}, marker)
    except FileExistsError as exc:
        raise ValueError(
            "test already started; preserve outputs and do not retune"
        ) from exc
    router = Router.load(output / "router.json")
    dataset, adaptive, _ = _retriever(data_dir, cache_dir, "test", router)
    report = run_enterprise_baselines(
        data_dir, cache_dir, "test", warmup=warmup, repeats=repeats, progress=progress
    )
    measured = _MeasuredRouter(adaptive)
    result = benchmark_retriever(measured, dataset, warmup=warmup, repeats=repeats)
    per_query = _quality_rows(result["queries"], dataset)
    slices = slice_definitions(load_query_metadata(data_dir, "test"))
    report["methods"]["adaptive"] = {
        "config": frozen["selected"],
        "latency": result["latency"],
        "queries": result["queries"],
        "slices": {
            name: summarize(per_query, ids) for name, ids in slices.items() if ids
        },
    }
    details = measured.details[len(dataset.queries) * warmup :]
    first = details[: len(dataset.queries)]
    report["mode_counts"] = dict(Counter(d["mode"] for d in first))
    report["router_overhead"] = {
        name: latency_summary([d[name] for d in details])
        for name in ("feature_ms", "router_ms")
    }
    report["router_overhead"]["combined_ms"] = latency_summary(
        [d["feature_ms"] + d["router_ms"] for d in details]
    )
    oracle_rows = []
    for query in dataset.queries:
        sparse, dense = adaptive.score_arrays(query.text)
        hits = [
            adaptive.rank_scores(sparse, dense, weight, 10)
            for weight in router.state["weights"]
        ]
        qualities = [ndcg_at_k(h, dataset.qrels[query.query_id], 10) for h in hits]
        mode = int(training_labels([qualities])[0])
        oracle_rows.append(
            {
                "query_id": query.query_id,
                "text": query.text,
                "mode": MODES[mode],
                "metrics": evaluate_query(hits[mode], dataset.qrels[query.query_id]),
                "results": [asdict(h) for h in hits[mode]],
            }
        )
    oracle_metrics = _quality_rows(oracle_rows, dataset)
    report["methods"]["oracle"] = {
        "config": {
            "selection": "best of frozen three modes by binary nDCG@10; post-hoc upper bound only"
        },
        "latency": None,
        "queries": oracle_rows,
        "slices": {
            name: summarize(oracle_metrics, ids) for name, ids in slices.items() if ids
        },
    }
    fixed = {r["query_id"]: r for r in report["methods"]["score_fusion"]["queries"]}
    cases = []
    for row, decision in zip(result["queries"], first, strict=True):
        qid = row["query_id"]
        cases.append(
            {
                "query_id": qid,
                "text": row["text"],
                "mode": decision["mode"],
                "delta_ndcg@10": row["metrics"]["ndcg@10"]
                - fixed[qid]["metrics"]["ndcg@10"],
                "oracle_regret_ndcg@10": oracle_metrics[qid]["ndcg@10"]
                - row["metrics"]["ndcg@10"],
                "fixed_top3": fixed[qid]["results"][:3],
                "adaptive_top3": row["results"][:3],
            }
        )
    report["failure_analysis"] = {
        "improved": sum(c["delta_ndcg@10"] > 1e-12 for c in cases),
        "degraded": sum(c["delta_ndcg@10"] < -1e-12 for c in cases),
        "tied": sum(abs(c["delta_ndcg@10"]) <= 1e-12 for c in cases),
        "worst": sorted(cases, key=lambda c: (c["delta_ndcg@10"], c["query_id"]))[:10],
        "best": sorted(cases, key=lambda c: (-c["delta_ndcg@10"], c["query_id"]))[:10],
        "largest_oracle_regret": sorted(
            cases, key=lambda c: (-c["oracle_regret_ndcg@10"], c["query_id"])
        )[:10],
    }
    ablations = {
        name: report["methods"][name]["slices"]["non_synthetic"]
        for name in ("dense", "score_fusion", "adaptive", "oracle")
    }
    delta = ablations["adaptive"]["ndcg@10"] - ablations["score_fusion"]["ndcg@10"]
    report["ablation"] = {
        "slice": "non_synthetic",
        "methods": ablations,
        "adaptive_minus_fixed_ndcg@10": delta,
        "outcome": "improved"
        if delta > 1e-12
        else "degraded"
        if delta < -1e-12
        else "tied",
        "reporting": "all outcomes retained; no test-based retuning",
    }
    report["router_sha256"] = frozen["router_sha256"]
    report["conditions"]["retuned"] = (
        "router selected on train/validation only; baselines unchanged"
    )
    report["conditions"]["test_exposure"] = protocol["test_exposure"]
    report["conditions"]["significance"] = (
        "descriptive comparison; no claim of statistically significant improvement"
    )
    report["code"]["sha256"].update(_code_hashes())
    write_json(output / "test-report.json", report)
    return report

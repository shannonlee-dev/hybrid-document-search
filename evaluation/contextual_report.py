"""Export and verify compact DAT rankings without checkpoints or raw text."""

import hashlib
import json
from collections import defaultdict
from math import fsum, isclose, isfinite
from pathlib import Path

import numpy as np

from evaluation.data import file_sha256, load_evaluation_dataset
from evaluation.json_io import read_json, write_json
from evaluation.latency import latency_summary
from evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from fusion.contextual import FixedPolicy, relevance_weight
from retrievers.base import SearchResult

METRIC_FUNCTIONS = {
    "p@1": lambda hits, labels: precision_at_k(hits, labels, 1),
    "recall@10": lambda hits, labels: recall_at_k(hits, labels, 10),
    "mrr@10": lambda hits, labels: reciprocal_rank_at_k(hits, labels, 10),
    "ndcg@10": lambda hits, labels: ndcg_at_k(hits, labels, 10),
}
DATA_FILES = ("manifest.json", "corpus.jsonl", "queries_test.jsonl", "qrels_test.jsonl")


def ranking_record(
    dataset,
    query_id,
    ranking,
    *,
    run_id,
    method,
    alpha=None,
    judged=None,
    top1=None,
    probabilities=None,
    normalization="union_minmax",
):
    """Compute binary metrics from actual corpus indices and official judgments."""
    indices = np.asarray(ranking)
    if (
        indices.ndim != 1
        or not 0 < len(indices) <= 10
        or not np.issubdtype(indices.dtype, np.integer)
        or (indices < 0).any()
        or (indices >= len(dataset.documents)).any()
        or len(set(indices.tolist())) != len(indices)
    ):
        raise ValueError("invalid or duplicate ranking indices")
    ids = [dataset.documents[int(i)].document_id for i in indices]
    labels = dataset.qrels[query_id]
    hits = [SearchResult(doc_id, rank, 0.0) for rank, doc_id in enumerate(ids, 1)]
    if alpha is not None:
        FixedPolicy(alpha, normalization)
    if probabilities is not None:
        expected = relevance_weight(probabilities)
        if (
            not judged
            or alpha is None
            or not isclose(alpha, expected, abs_tol=1e-12, rel_tol=0)
        ):
            raise ValueError("judge probabilities differ from selected alpha")
    return {
        "run_id": run_id,
        "query_id": query_id,
        "method": method,
        **{name: fn(hits, labels) for name, fn in METRIC_FUNCTIONS.items()},
        "top10_document_ids": ids,
        "relevant_rank_top10": next(
            (r for r, d in enumerate(ids, 1) if labels.get(d, 0) > 0), None
        ),
        "selected_alpha_dense": float(alpha) if alpha is not None else None,
        "selected_mode": "judge_top1"
        if judged
        else "fixed_top1_agreement"
        if judged is False and method != "dat_fixed"
        else "fixed"
        if alpha is not None
        else None,
        "normalization": normalization if alpha is not None else None,
        "top1_document_ids_dense_sparse": list(top1) if top1 is not None else None,
        "judge_probabilities_dense_sparse": list(probabilities)
        if probabilities is not None
        else None,
        "oracle_alpha_dense": None,
        "oracle_mode": None,
    }


def _validate_dat_routing(row, document_ids, fixed_policy):
    """Bind routing metadata to the method, channel agreement and selected policy."""
    top1 = row.get("top1_document_ids_dense_sparse")
    if not top1 or len(top1) != 2 or any(d not in document_ids for d in top1):
        raise ValueError("invalid top1 metadata")
    is_fixed = row["method"] == "dat_fixed"
    agreement = top1[0] == top1[1]
    expected_mode = (
        "fixed" if is_fixed else "fixed_top1_agreement" if agreement else "judge_top1"
    )
    if row.get("selected_mode") != expected_mode:
        raise ValueError("selected mode contradicts method or top1 agreement")
    alpha = row.get("selected_alpha_dense")
    if alpha is None:
        raise ValueError("DAT routing requires selected alpha")
    FixedPolicy(alpha, row.get("normalization"))
    probabilities = row.get("judge_probabilities_dense_sparse")
    if is_fixed or agreement:
        if probabilities is not None:
            raise ValueError("Fixed routing must not contain judge probabilities")
        if (
            not isclose(alpha, fixed_policy.alpha, abs_tol=1e-12, rel_tol=0)
            or row["normalization"] != fixed_policy.normalization
        ):
            raise ValueError("Fixed routing differs from selected Fixed policy")
    elif (
        probabilities is None
        or row["normalization"] != "union_minmax"
        or not isclose(alpha, relevance_weight(probabilities), abs_tol=1e-12, rel_tol=0)
    ):
        raise ValueError(
            "judge probabilities or normalization differ from selected alpha"
        )


def _selected_fixed_policy(methods, selection):
    if {"dat_fixed", "dat"}.intersection(methods):
        if not {"alpha", "normalization"}.issubset(selection):
            raise ValueError(
                "DAT requires selected Fixed policy alpha and normalization"
            )
        return FixedPolicy(selection["alpha"], selection["normalization"])
    return FixedPolicy()


def validate_records(
    records,
    dataset,
    query_ids,
    methods,
    *,
    method_runs=None,
    fixed_policy=FixedPolicy(),
):
    """Validate coverage, metrics, policy-bound routing and matched channel metadata."""
    by_id = {d.document_id: i for i, d in enumerate(dataset.documents)}
    expected = {(q, m) for q in query_ids for m in methods}
    if not query_ids or len(set(query_ids)) != len(query_ids):
        raise ValueError("unique nonempty query coverage required")
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("unique nonempty method coverage required")
    if method_runs is not None and set(method_runs) != set(methods):
        raise ValueError("run mapping must cover every method")
    seen, runs, values = set(), defaultdict(set), defaultdict(list)
    for row in records:
        key = row["query_id"], row["method"]
        if key in seen:
            raise ValueError("duplicate query/method record")
        if key not in expected:
            raise ValueError("unexpected query/method coverage")
        seen.add(key)
        run_id = row["run_id"]
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("nonempty run identity required")
        runs[key[1]].add(run_id)
        if method_runs is not None and method_runs[key[1]] != run_id:
            raise ValueError("method run identity differs from protocol")
        ids = row["top10_document_ids"]
        if any(d not in by_id for d in ids):
            raise ValueError("ranking contains unknown document")
        if key[1] in ("dat_fixed", "dat"):
            _validate_dat_routing(row, by_id, fixed_policy)
        mode = row.get("selected_mode")
        top1 = row.get("top1_document_ids_dense_sparse")
        probs = row.get("judge_probabilities_dense_sparse")
        if mode in ("judge_top1", "fixed_top1_agreement"):
            if not top1 or len(top1) != 2 or any(d not in by_id for d in top1):
                raise ValueError("invalid top1 metadata")
            if (mode == "judge_top1") != (top1[0] != top1[1]):
                raise ValueError("routing contradicts top1 agreement")
            if mode == "judge_top1" and probs is None:
                raise ValueError("judge probabilities required")
            if mode == "fixed_top1_agreement" and probs is not None:
                raise ValueError("agreement must not invent probabilities")
        check = ranking_record(
            dataset,
            key[0],
            [by_id[d] for d in ids],
            run_id=row["run_id"],
            method=key[1],
            alpha=row.get("selected_alpha_dense"),
            judged=True
            if mode == "judge_top1"
            else False
            if mode == "fixed_top1_agreement"
            else None,
            top1=top1,
            probabilities=probs,
            normalization=row.get("normalization") or "union_minmax",
        )
        for name in (*METRIC_FUNCTIONS, "relevant_rank_top10"):
            if row[name] != check[name]:
                raise ValueError(f"metric differs from qrels: {name}")
        values[key[1]].append(row)
    if seen != expected:
        raise ValueError("incomplete query/method coverage")
    if any(len(ids) != 1 for ids in runs.values()):
        raise ValueError("mixed run identities within a method")
    if {"dat_fixed", "dat"}.issubset(methods):
        indexed = {(r["query_id"], r["method"]): r for r in records}
        for query_id in query_ids:
            fixed, adaptive = [indexed[query_id, m] for m in ("dat_fixed", "dat")]
            top1 = fixed["top1_document_ids_dense_sparse"]
            if top1 != adaptive["top1_document_ids_dense_sparse"]:
                raise ValueError("matched DAT/Fixed top1 metadata differs")
            if (
                top1[0] == top1[1]
                and fixed["top10_document_ids"] != adaptive["top10_document_ids"]
            ):
                raise ValueError(
                    "Top10 differs between DAT and Fixed on top1 agreement"
                )
    return {
        method: {
            name: fsum(r[name] for r in rows) / len(rows) for name in METRIC_FUNCTIONS
        }
        for method, rows in values.items()
    }


def ordered_ids_sha256(ids):
    """Hash UTF-8 IDs joined by LF, without a trailing LF."""
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def compact_encoding(encoding):
    """Retain frozen settings and vector identity without copying full ID arrays."""
    return {
        **encoding,
        "splits": {
            split: {
                **{k: v for k, v in details.items() if k != "query_ids"},
                **(
                    {
                        "queries": len(details["query_ids"]),
                        "query_ids_sha256": ordered_ids_sha256(details["query_ids"]),
                    }
                    if "query_ids" in details
                    else {}
                ),
            }
            for split, details in encoding.get("splits", {}).items()
        },
    }


def _validate_baseline_audit(audit, summary, query_count):
    methods = {"bm25", "dense", "score_fusion"}
    if (
        audit.get("all_three_main_baselines_reproduced") is not True
        or audit.get("queries") != query_count
        or set(audit["canonical_metrics"]) != methods
        or set(audit["replayed_metrics"]) != methods
    ):
        raise ValueError("invalid canonical baseline audit")
    for method in methods:
        if audit["replayed_metrics"][method] != summary[method]:
            raise ValueError("baseline audit differs from rankings")
        for name, value in summary[method].items():
            if not isclose(
                value,
                audit["canonical_metrics"][method][name],
                abs_tol=1e-12,
                rel_tol=0,
            ):
                raise ValueError("baseline differs from canonical metrics")


def save_handoff(
    output,
    data_directory,
    records,
    *,
    selection,
    protocol,
    environment=None,
    samples=None,
    baseline_audit=None,
):
    """Write a compact bundle with method-specific runs and independently auditable time."""
    output, data_directory = Path(output), Path(data_directory)
    if output.exists():
        raise FileExistsError(f"run already exists: {output}")
    dataset = load_evaluation_dataset(data_directory, "test", verify_all_splits=False)
    manifest = read_json(data_directory / "manifest.json")
    query_ids = manifest["primary_slice_query_ids"]
    if list(dict.fromkeys(r["query_id"] for r in records)) != query_ids:
        raise ValueError("handoff must cover the official primary query IDs in order")
    methods = list(dict.fromkeys(r["method"] for r in records))
    method_runs = {r["method"]: r["run_id"] for r in records}
    summary = validate_records(
        records,
        dataset,
        query_ids,
        methods,
        method_runs=method_runs,
        fixed_policy=_selected_fixed_policy(methods, selection),
    )
    results = {"queries": len(query_ids), "methods": summary}
    if baseline_audit is not None:
        _validate_baseline_audit(baseline_audit, summary, len(query_ids))
        results["baseline_audit"] = baseline_audit
    if "encoding" in protocol:
        protocol = {**protocol, "encoding": compact_encoding(protocol["encoding"])}
    if "encoding" in selection:
        selection = {**selection, "encoding": compact_encoding(selection["encoding"])}
    protocol = {
        **protocol,
        "schema_version": 2,
        "queries": len(query_ids),
        "query_ids_sha256": ordered_ids_sha256(query_ids),
        "query_ids_hash_format": "UTF-8 IDs joined by LF, no trailing LF",
        "methods": methods,
        "method_runs": method_runs,
        "rows": len(records),
        "dataset": {
            "id": manifest["dataset_id"],
            "revision": manifest["dataset_revision"],
            "corpus_documents": len(dataset.documents),
            "query_scope": "official primary non_synthetic test",
        },
        "input_sha256": {
            name: file_sha256(data_directory / name) for name in DATA_FILES
        },
        "selection": selection,
        "oracle": "not computed; Phase 5 analysis excluded",
    }
    if environment is not None:
        protocol["environment"] = environment
    if samples is not None:
        timed_methods = [m for m in methods if m in {s["method"] for s in samples}]
        if set(timed_methods) != {"dat_fixed", "dat"}:
            raise ValueError("matched DAT/Fixed timing samples required")
        passes = environment["passes"]
        results["latency"] = validate_timings(
            samples, query_ids, timed_methods, repeats=passes
        )
        protocol["timing"] = {
            "method_runs": {m: method_runs[m] for m in timed_methods},
            "passes": passes,
            "samples": len(samples),
            "excluded_passes": [0],
        }
    output.mkdir(parents=True)
    (output / "per-query.jsonl").write_text(
        "".join(
            json.dumps(r, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
            for r in records
        ),
        encoding="utf-8",
    )
    write_json(output / "results.json", results)
    if samples is not None:
        (output / "latency-samples.json").write_text(
            "[\n"
            + ",\n".join(
                json.dumps(s, allow_nan=False, separators=(",", ":")) for s in samples
            )
            + "\n]\n",
            encoding="utf-8",
        )
    protocol["artifact_sha256"] = {
        p.name: file_sha256(p) for p in output.iterdir() if p.is_file()
    }
    write_json(output / "protocol.json", protocol)


def verify_handoff(data_directory, output):
    """Recompute all ranking metrics and raw timing aggregates, without models or GPU."""
    data_directory, output = Path(data_directory), Path(output)
    protocol = read_json(output / "protocol.json")
    if protocol["schema_version"] != 2:
        raise ValueError("unsupported handoff schema")
    artifacts = {"per-query.jsonl", "results.json"}
    if "timing" in protocol:
        artifacts.add("latency-samples.json")
    if set(protocol["artifact_sha256"]) != artifacts:
        raise ValueError("artifact hash coverage differs")
    for name, digest in protocol["artifact_sha256"].items():
        if file_sha256(output / name) != digest:
            raise ValueError(f"artifact hash mismatch: {name}")
    for name in DATA_FILES:
        if file_sha256(data_directory / name) != protocol["input_sha256"].get(name):
            raise ValueError(f"dataset hash mismatch: {name}")
    dataset = load_evaluation_dataset(data_directory, "test", verify_all_splits=False)
    expected = read_json(data_directory / "manifest.json")["primary_slice_query_ids"]
    if (
        len(expected) != len(set(expected))
        or protocol["queries"] != len(expected)
        or protocol["query_ids_sha256"] != ordered_ids_sha256(expected)
    ):
        raise ValueError("handoff primary query identity differs")
    records = [
        json.loads(line)
        for line in (output / "per-query.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if len(records) != protocol["rows"]:
        raise ValueError("row count differs")
    if list(dict.fromkeys(r["query_id"] for r in records)) != expected:
        raise ValueError("handoff must cover the official primary query IDs in order")
    actual = validate_records(
        records,
        dataset,
        expected,
        protocol["methods"],
        method_runs=protocol["method_runs"],
        fixed_policy=_selected_fixed_policy(protocol["methods"], protocol["selection"]),
    )
    results = read_json(output / "results.json")
    if results["queries"] != len(expected) or results["methods"] != actual:
        raise ValueError("aggregate metrics differ")
    if "baseline_audit" in results:
        _validate_baseline_audit(results["baseline_audit"], actual, len(expected))
    checked = {
        "queries": len(expected),
        "rows": len(records),
        "methods": actual,
        "verified": True,
    }
    if "timing" in protocol:
        timing = protocol["timing"]
        methods = list(timing["method_runs"])
        if (
            set(methods) != {"dat_fixed", "dat"}
            or timing["excluded_passes"] != [0]
            or timing["passes"] != protocol["environment"]["passes"]
            or any(
                protocol["method_runs"].get(m) != run
                for m, run in timing["method_runs"].items()
            )
        ):
            raise ValueError("timing methods, runs or passes differ")
        samples = read_json(output / "latency-samples.json")
        if len(samples) != timing["samples"]:
            raise ValueError("timing sample count differs")
        summary = validate_timings(samples, expected, methods, repeats=timing["passes"])
        if results.get("latency") != summary:
            raise ValueError("latency metrics differ from raw samples")
        checked.update(timing_samples=len(samples), latency=summary)
    elif "latency" in results:
        raise ValueError("latency requires raw timing samples")
    return checked


def combine_handoffs(data_directory, baseline, live, output):
    """Combine verified canonical and live bundles while preserving original run IDs."""
    baseline, live = Path(baseline), Path(live)
    for path in (baseline, live):
        verify_handoff(data_directory, path)
    base_protocol, live_protocol = [
        read_json(p / "protocol.json") for p in (baseline, live)
    ]
    if (
        set(base_protocol["methods"]) != {"bm25", "dense", "score_fusion"}
        or set(live_protocol["methods"]) != {"dat_fixed", "dat"}
        or "timing" not in live_protocol
    ):
        raise ValueError("canonical three methods and matched live DAT/Fixed required")
    if set(base_protocol["method_runs"].values()) & set(
        live_protocol["method_runs"].values()
    ):
        raise ValueError("canonical and DAT must retain distinct run identities")
    records = [
        json.loads(line)
        for p in (baseline, live)
        for line in (p / "per-query.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    primary = read_json(Path(data_directory) / "manifest.json")[
        "primary_slice_query_ids"
    ]
    methods = [*base_protocol["methods"], *live_protocol["methods"]]
    index = {(r["query_id"], r["method"]): r for r in records}
    records = [index[q, m] for q in primary for m in methods]
    results = read_json(baseline / "results.json")
    save_handoff(
        output,
        data_directory,
        records,
        selection=live_protocol["selection"],
        protocol={
            "scope": "canonical main baselines and matched single-query DAT evaluation",
            "runs": {
                run_id: protocol
                for protocol in (base_protocol, live_protocol)
                for run_id in dict.fromkeys(protocol["method_runs"].values())
            },
            "conversion": "Verified existing rankings and times; no new model inference",
        },
        environment=live_protocol["environment"],
        samples=read_json(live / "latency-samples.json"),
        baseline_audit=results.get("baseline_audit"),
    )


def validate_timings(samples, query_ids, methods, *, repeats):
    """Recompute latency after checking every timed request and its stage sum."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
        raise ValueError("at least one quality and one measured pass required")
    stages = ("encode", "retrieve", "judge", "fusion", "total")
    expected = {(q, m, r) for q in query_ids for m in methods for r in range(repeats)}
    seen = set()
    for row in samples:
        key = row["query_id"], row["method"], row["repeat"]
        if key in seen:
            raise ValueError("duplicate timed request")
        if key not in expected:
            raise ValueError("unexpected timing coverage")
        seen.add(key)
        if any(
            isinstance(row[s], bool) or not isfinite(row[s]) or row[s] < 0
            for s in stages
        ):
            raise ValueError("timing stages must be finite and nonnegative")
        if not isclose(
            fsum(row[s] for s in stages[:-1]), row["total"], abs_tol=1e-6, rel_tol=1e-9
        ):
            raise ValueError("timing stage sum differs from request total")
    if seen != expected:
        raise ValueError("incomplete timing coverage")
    measured = [s for s in samples if s["repeat"] > 0]
    return {
        method: {
            stage: latency_summary(
                [s[stage] for s in measured if s["method"] == method]
            )
            for stage in stages
        }
        for method in methods
    }

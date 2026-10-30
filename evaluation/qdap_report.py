"""Convert a frozen QDAP JSONL handoff to primary-slice enterprise reports."""

from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path

from data.bccard import load_query_metadata
from evaluation.benchmark import _code_provenance, _environment
from evaluation.data import load_evaluation_dataset
from evaluation.enterprise import query_metrics, slice_definitions, summarize
from evaluation.json_io import read_json
from evaluation.metrics import evaluate_query
from evaluation.qdap_benchmark import rows, sha
from retrievers.base import SearchResult

METHOD_NAMES = {
    "BM25": "qdap_bm25",
    "Dense": "qdap_dense_fp64",
    "Existing Fixed": "qdap_fixed_fp64",
    "Existing Router": "qdap_router_fp64",
    "Paper Fixed": "paper_fixed",
    "QDAP-S": "qdap_s",
    "QDAP-L": "qdap_l",
    "Oracle alpha": "oracle_alpha",
}
CANONICAL_METHOD_NAMES = {
    "BM25": "bm25",
    "Dense": "dense",
    "Existing Fixed": "score_fusion",
}
METRICS = ("p@1", "recall@10", "mrr@10", "ndcg@10")
ROOT = Path(__file__).resolve().parents[1]
RELEVANCE = "binary metrics use relevance > 0; graded nDCG uses gains 1-3"


def enterprise_report(inputs, source, *, canonical_summary=None):
    """Export actual primary rankings; recompute metrics and verify source identity.

    No model inference, search, or modification of the source JSONL is needed.
    Scores and latency absent from JSONL stay null, rather than being invented.
    """
    inputs, source = Path(inputs), Path(source)
    data = inputs / "data/processed/bccard"
    manifest = read_json(data / "manifest.json")
    provenance = read_json(source.parent / "per-query-manifest.json")
    selection_path = source.parent / "selection.json"
    if sha(source) != provenance["sha256"]:
        raise ValueError("per-query file hash mismatch")
    if sha(selection_path) != provenance["selection_sha256"]:
        raise ValueError("selection hash mismatch")
    expected = {
        Path(p).name: digest for p, digest in provenance["input_sha256"].items()
    }
    for name in (
        "manifest.json",
        "corpus.jsonl",
        "queries_test.jsonl",
        "qrels_test.jsonl",
    ):
        if expected.get(name) != sha(data / name):
            raise ValueError(f"dataset identity mismatch: {name}")
    dataset = load_evaluation_dataset(data, "test")
    primary_ids = manifest["primary_slice_query_ids"]
    if len(primary_ids) != len(set(primary_ids)) or not primary_ids:
        raise ValueError("unique nonempty primary query IDs required")
    metadata = load_query_metadata(data, "test")
    if set(primary_ids) - set(metadata):
        raise ValueError("primary query IDs absent from dataset")
    primary_set = set(primary_ids)
    slices = {
        name: [q for q in ids if q in primary_set]
        for name, ids in slice_definitions(metadata).items()
    }
    selections = read_json(selection_path)
    protocol_path = source.parent / "protocol.json"
    protocol = read_json(protocol_path)
    if selections.get("protocol_sha256") is not None and selections[
        "protocol_sha256"
    ] != sha(protocol_path):
        raise ValueError("protocol hash mismatch")
    # These identities describe the frozen source run, never the recipient's cache.
    encoding = deepcopy(provenance.get("encoding", protocol.get("encoding")))
    if not encoding:
        raise ValueError(
            "saved encoding provenance required in per-query manifest or protocol"
        )
    recorded = {}
    for hashes in (
        protocol.get("input_sha256", {}),
        selections.get("evaluation_input_sha256", {}),
    ):
        for path, digest in hashes.items():
            name = Path(path).name
            if (
                name != "encoding.json"
                and name != "document-vectors.npy"
                and not name.startswith("query-vectors-")
            ):
                continue
            if name in recorded and recorded[name] != digest:
                raise ValueError(f"conflicting saved encoding identity: {name}")
            recorded[name] = digest
    if (
        provenance.get("encoding_record_sha256") is not None
        and recorded.get("encoding.json") != provenance["encoding_record_sha256"]
    ):
        raise ValueError("saved encoding record identity mismatch")
    for name, digest in recorded.items():
        if name != "encoding.json" and encoding["vector_sha256"].get(name) != digest:
            raise ValueError(f"saved vector identity mismatch: {name}")
    if (
        encoding.get("dataset_manifest_sha256") is not None
        and encoding["dataset_manifest_sha256"] != dataset.provenance["manifest_sha256"]
    ):
        raise ValueError("encoding dataset identity mismatch")
    grouped, run_id = _read_methods(source, dataset, primary_ids, METHOD_NAMES)
    report = {
        "schema_version": "1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "split": "test",
            **dataset.provenance,
            "query_ids": primary_ids,
            "query_count": len(primary_ids),
        },
        "conditions": {
            "top_k": 10,
            "relevance": RELEVANCE,
            "query_scope": "primary non_synthetic only",
            "quality_run": run_id,
            "official_baselines": list(CANONICAL_METHOD_NAMES.values()),
            "latency_scope": "not exported; JSONL contains no timing samples",
            "retrieval_scores": "not recorded in source JSONL; score is null",
            "test_tuning": False,
        },
        "encoding": {
            k: encoding[k]
            for k in ("model", "dtype", "max_seq_length", "vector_sha256")
        },
        "code": _code_provenance(),
        "environment": _environment(),
        "source": {
            "file": str(source),
            "sha256": sha(source),
            "selection_sha256": sha(selection_path),
            "protocol_sha256": sha(protocol_path),
            "per_query_manifest_sha256": sha(source.parent / "per-query-manifest.json"),
            "encoding_record_sha256": recorded.get("encoding.json"),
            "code_provenance": _source_provenance(selections, source.parent),
        },
        "slices": {name: len(ids) for name, ids in slices.items() if ids},
        "methods": {},
    }
    report["code"]["sha256"].update(
        {
            "evaluation/qdap_report.py": sha(Path(__file__)),
            "scripts/export_qdap_results.py": sha(
                Path("scripts/export_qdap_results.py")
            ),
        }
    )
    for method, by_id in grouped.items():
        per_query = {q: row["metrics"] for q, row in by_id.items()}
        config = {
            "source_method": method,
            "source_run_id": run_id,
            "baseline_role": "qdap_protocol",
            "score_precision": "float64 from frozen FP32 vectors",
        }
        if method in ("Paper Fixed", "QDAP-S", "QDAP-L", "Oracle alpha"):
            config |= {
                "candidate_scope": "Dense/BM25 Top100 union",
                "normalization": "per-query union min-max",
            }
        if method == "Paper Fixed":
            config["alpha_dense"] = selections["paper_fixed_alpha"]
        elif method in ("QDAP-S", "QDAP-L"):
            config |= {
                "prediction": selections["selected"][method[-1]],
                "alpha": "Dense coefficient after union-minmax normalization",
            }
        elif method == "Existing Fixed":
            config |= {"bm25_weight": 0.05, "normalization": "per-query corpus maximum"}
        elif method == "Oracle alpha":
            config["selection"] = (
                "post-hoc nDCG argmax; smallest alpha on ties; diagnostic only"
            )
        report["methods"][METHOD_NAMES[method]] = {
            "config": config,
            "latency": None,
            "queries": [by_id[q] for q in primary_ids],
            "slices": {
                name: summarize(per_query, ids) for name, ids in slices.items() if ids
            },
        }
    _add_canonical_baselines(
        report, source.parent, canonical_summary, dataset, slices, primary_ids
    )
    return report


def _read_methods(source, dataset, primary_ids, method_names):
    grouped = defaultdict(dict)
    run_ids = set()
    document_ids = {d.document_id for d in dataset.documents}
    primary_set = set(primary_ids)
    for row in rows(source):
        method, query_id = row["method"], row["query_id"]
        if method not in method_names or query_id not in primary_set:
            raise ValueError("unknown method or query outside primary slice")
        if query_id in grouped[method]:
            raise ValueError("duplicate method/query row")
        run_ids.add(row["run_id"])
        ids = row["top10_document_ids"]
        if (
            not 0 < len(ids) <= 10
            or len(ids) != len(set(ids))
            or set(ids) - document_ids
        ):
            raise ValueError("invalid Top10 document IDs")
        hits = [
            {"document_id": d, "rank": i, "score": None} for i, d in enumerate(ids, 1)
        ]
        # Metric helpers only inspect IDs/ranks; original retrieval scores are unavailable.
        results = [SearchResult(**hit) for hit in hits]
        judged = dataset.qrels[query_id]
        metrics = evaluate_query(results, judged) | query_metrics(hits, judged)
        metrics["mrr@10"] = metrics["rr@10"]
        for metric in ("p@1", "recall@10", "mrr@10", "ndcg@10"):
            if not isfinite(row[metric]) or abs(metrics[metric] - row[metric]) > 1e-12:
                raise ValueError(f"saved metric mismatch: {method}/{query_id}/{metric}")
        # Preserve the saved values exactly after independent recomputation.
        metrics.update({metric: row[metric] for metric in METRICS})
        metrics["rr@10"] = metrics["mrr@10"]
        rank = next((i for i, d in enumerate(ids, 1) if judged.get(d, 0) > 0), None)
        if rank != row["relevant_rank_top10"]:
            raise ValueError("saved relevant rank mismatch")
        grouped[method][query_id] = {
            "query_id": query_id,
            "metrics": metrics,
            "results": hits,
            "latency": None,
            **{
                k: row.get(k)
                for k in (
                    "selected_alpha_dense",
                    "oracle_alpha_dense",
                    "selected_mode",
                    "bm25_weight",
                )
            },
        }
    if not grouped or len(run_ids) != 1:
        raise ValueError("one nonempty source run required")
    for method, by_id in grouped.items():
        if set(by_id) != primary_set:
            raise ValueError(f"incomplete primary query set: {method}")
    return grouped, next(iter(run_ids))


def _source_provenance(selection, directory):
    frozen = selection.get("source_sha256", {})
    current = {}
    for name in frozen:
        # Old selections contain both repo-relative and absolute worktree paths.
        path = Path(name)
        local = ROOT / (Path(*path.parts[-2:]) if path.is_absolute() else path)
        current[name] = sha(local) if local.is_file() else None
    result = {
        "scope": "saved-artifact metric replay; no source parity or inference claim",
        "frozen_experiment_source_sha256": frozen,
        "current_source_sha256": current,
        "source_hash_mismatches": [
            name for name in frozen if frozen[name] != current[name]
        ],
    }
    parity_path = directory / "score-parity.json"
    if parity_path.is_file():
        parity = read_json(parity_path)
        result["historical_score_parity"] = {
            "sha256": sha(parity_path),
            "code_sha256": parity["code_sha256"],
            "preserved_selection_sha256": parity["preserved_selection_sha256"],
            "scope": "historical score-path validation; does not certify current code",
        }
    return result


def _add_canonical_baselines(report, directory, summary_path, dataset, slices, ids):
    audit_path = directory / "baseline-audit.json"
    audit = read_json(audit_path)
    source = directory / "main-baseline-per-query.jsonl"
    summary_path = Path(summary_path or audit["canonical_summary"])
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    if sha(source) != audit["per_query_sha256"]:
        raise ValueError("canonical per-query file hash mismatch")
    if sha(summary_path) != audit["canonical_summary_sha256"]:
        raise ValueError("canonical summary hash mismatch")
    canonical = read_json(summary_path)
    if canonical["dataset"]["split"] != "test":
        raise ValueError("canonical Test baseline required")
    for name, digest in canonical["dataset"]["input_sha256"].items():
        if dataset.provenance["input_sha256"].get(name) != digest:
            raise ValueError(f"canonical dataset identity mismatch: {name}")
    grouped, run_id = _read_methods(source, dataset, ids, CANONICAL_METHOD_NAMES)
    if set(grouped) != set(CANONICAL_METHOD_NAMES) or run_id != audit["run_id"]:
        raise ValueError("three canonical methods from the audited run required")
    for method, name in CANONICAL_METHOD_NAMES.items():
        by_id = grouped[method]
        per_query = {q: row["metrics"] for q, row in by_id.items()}
        actual = summarize(per_query, ids)
        expected = canonical["methods"][name]["slices"]["non_synthetic"]
        if expected["query_count"] != len(ids):
            raise ValueError("canonical primary query count mismatch")
        for metric in METRICS:
            if (
                not isfinite(expected[metric])
                or abs(actual[metric] - expected[metric]) > 1e-12
            ):
                raise ValueError(f"canonical metric mismatch: {name}/{metric}")
        report["methods"][name] = {
            "config": {
                **canonical["methods"][name]["config"],
                "source_method": method,
                "source_run_id": run_id,
                "baseline_role": "main_canonical",
                "retrieval_path": "main CPU baseline; FAISS IndexFlatIP FP32 Dense",
                "canonical_encoding": canonical["encoding"],
                "replay_vector_sha256": audit["current_vector_sha256"],
            },
            "latency": None,
            "queries": [by_id[q] for q in ids],
            "slices": {
                name: summarize(per_query, queries)
                for name, queries in slices.items()
                if queries
            },
        }
    report["conditions"]["canonical_baseline_run"] = run_id
    report["source"]["canonical_baselines"] = {
        "file": str(source),
        "sha256": sha(source),
        "audit_sha256": sha(audit_path),
        "canonical_summary": str(summary_path),
        "canonical_summary_sha256": sha(summary_path),
        "scope": "reuse of saved main replay; four metrics verified against canonical",
    }


def align_reference_report(reference, target):
    """Copy a compatible enterprise report onto the exact exported primary IDs.

    PR #37 hashes all per-method query IDs, so its 1,000-query reports must be
    restricted to the same ordered 721 IDs before comparing a primary-only run.
    """
    for key in (
        "split",
        "dataset_id",
        "dataset_revision",
        "manifest_sha256",
        "corpus_size",
    ):
        if reference["dataset"].get(key) != target["dataset"].get(key):
            raise ValueError(f"reference dataset identity mismatch: {key}")
    split = target["dataset"]["split"]
    for name in ("corpus.jsonl", f"queries_{split}.jsonl", f"qrels_{split}.jsonl"):
        if reference["dataset"]["input_sha256"].get(name) != target["dataset"][
            "input_sha256"
        ].get(name):
            raise ValueError(f"reference dataset identity mismatch: {name}")
    for key in ("top_k", "relevance"):
        if reference["conditions"].get(key) != target["conditions"].get(key):
            raise ValueError(f"reference protocol identity mismatch: {key}")
    report = deepcopy(reference)
    ids = target["dataset"]["query_ids"]
    report["dataset"] |= {"query_ids": ids, "query_count": len(ids)}
    report["conditions"]["query_scope"] = "primary non_synthetic only"
    report["slices"] = {"all": len(ids), "non_synthetic": len(ids)}
    for method, data in report["methods"].items():
        by_id = {row["query_id"]: row for row in data["queries"]}
        if len(by_id) != len(data["queries"]) or set(ids) - set(by_id):
            raise ValueError(f"missing or duplicate reference query IDs: {method}")
        data["queries"] = [by_id[q] for q in ids]
        metrics = {q: by_id[q]["metrics"] for q in ids}
        data["slices"] = {
            name: summarize(metrics, ids) for name in ("all", "non_synthetic")
        }
        data["latency"] = None  # The original aggregate covers a different query set.
    return report

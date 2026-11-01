"""Compact query-encoder provenance and verification without model dependencies."""

import json
from collections import defaultdict
from math import fsum, isclose
from pathlib import Path

from evaluation.data import file_sha256, load_evaluation_dataset
from evaluation.json_io import read_json, write_json
from evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from retrievers.base import SearchResult

METRICS = ("p@1", "recall@10", "mrr@10", "ndcg@10")
DATA_FILES = ("manifest.json", "corpus.jsonl", "queries_test.jsonl", "qrels_test.jsonl")


def source_hashes():
    """Identify the implementation used to select and evaluate this run."""
    root = Path(__file__).resolve().parents[1]
    return {
        name: file_sha256(root / name)
        for name in (
            "evaluation/query_encoder.py",
            "evaluation/query_encoder_report.py",
            "evaluation/query_encoder_training.py",
            "evaluation/metrics.py",
            "scripts/evaluate_query_encoder.py",
            "scripts/train_query_encoder.py",
        )
    }


def model_identity(directory):
    """Hash local model weights, tokenizer, pooling and configuration by relative path."""
    directory = Path(directory)
    files = {
        str(path.relative_to(directory)): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix in (".bin", ".safetensors", ".json", ".model")
    }
    if not any(name.endswith((".bin", ".safetensors")) for name in files):
        raise ValueError("local model snapshot has no weight files")
    return files


def training_run_provenance(run_id, device):
    """Capture code and runtime identity before a new training run starts."""
    import importlib.metadata
    from datetime import UTC, datetime

    import torch

    from evaluation.benchmark import _code_provenance, _environment

    code = _code_provenance()
    hashes = source_hashes()
    code["sha256"].update(hashes)
    environment = _environment()
    for name in ("torch", "transformers"):
        environment["packages"][name] = importlib.metadata.version(name)
    environment.update(
        {
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0)
            if str(device).startswith("cuda")
            else None,
            "torch_threads": torch.get_num_threads(),
        }
    )
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code": code,
        "source_sha256": hashes,
        "environment": environment,
    }


def encoder_provenance(directory):
    """Preserve archived training choices without embedding local machine paths."""
    directory = Path(directory)
    frozen = read_json(directory / "frozen.json")
    checkpoint_hash = file_sha256(directory / "selected-last-block.pt")
    if checkpoint_hash != frozen["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint checksum mismatch")
    result = {**frozen, "checkpoint_sha256": checkpoint_hash}
    protocol_path = directory / "protocol.json"
    if protocol_path.is_file():
        protocol = read_json(protocol_path)
        result["training"] = {
            key: value
            for key, value in protocol.items()
            if key not in ("model", "input_sha256")
        }
        result["training_protocol_sha256"] = file_sha256(protocol_path)
        result["training_input_sha256"] = {
            Path(key).name: value
            for key, value in protocol.get("input_sha256", {}).items()
        }
        history = frozen.get("history", [])
        if history:
            key = (
                "validation_primary_ndcg"
                if "validation_primary_ndcg" in history[0]
                else "validation_ndcg"
            )
            winner = max(history, key=lambda entry: entry[key])
            if winner["step"] != frozen["selected_step"]:
                raise ValueError("checkpoint differs from validation history choice")
    return result


def method_configs(frozen):
    """Describe validation-selected policies and optional predeclared shared-alpha controls."""
    methods = {
        f"{name} + Fixed": {
            **choice,
            "encoder": name,
            "selection_policy": "validation_selected",
        }
        for name, choice in frozen["models"].items()
    }
    if frozen.get("shared_alpha_dense") is not None:
        alpha = frozen["shared_alpha_dense"]
        grid = frozen["alphas"]
        index = min(range(len(grid)), key=lambda i: abs(grid[i] - alpha))
        if not isclose(grid[index], alpha, abs_tol=1e-12, rel_tol=0):
            raise ValueError("shared alpha must belong to the frozen validation grid")
        for name in frozen["models"]:
            methods[f"{name} + Fixed alpha={alpha:g}"] = {
                "encoder": name,
                "alpha_dense": alpha,
                "alpha_index": index,
                "selection_policy": "shared_alpha",
            }
    return methods


def _averages(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["method"]].append(row)
    return {
        method: {key: fsum(row[key] for row in rows) / len(rows) for key in METRICS}
        for method, rows in grouped.items()
    }


def write_manifest(data_dir, run_dir, records):
    """Bind result bytes and frozen choices to dataset identity and aggregates."""
    import hashlib

    data_dir, run_dir = Path(data_dir), Path(run_dir)
    ids = read_json(data_dir / "manifest.json")["primary_slice_query_ids"]
    write_json(
        run_dir / "per-query-manifest.json",
        {
            "schema_version": 1,
            "run_id": read_json(run_dir / "frozen.json")["run_id"],
            "queries": len(ids),
            "rows": len(records),
            "methods": list(dict.fromkeys(row["method"] for row in records)),
            "query_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "sha256": file_sha256(run_dir / "per-query.jsonl"),
            "selection_sha256": file_sha256(run_dir / "frozen.json"),
            "input_sha256": {name: file_sha256(data_dir / name) for name in DATA_FILES},
            "aggregates": _averages(records),
        },
    )


def verify_results(data_dir, run_dir):
    """Recompute every metric and relevant rank using only prepared data and JSON."""
    import hashlib

    data_dir, run_dir = Path(data_dir), Path(run_dir)
    provenance = read_json(run_dir / "per-query-manifest.json")
    frozen = read_json(run_dir / "frozen.json")
    if file_sha256(run_dir / "per-query.jsonl") != provenance["sha256"]:
        raise ValueError("per-query checksum mismatch")
    if file_sha256(run_dir / "frozen.json") != provenance["selection_sha256"]:
        raise ValueError("frozen selection checksum mismatch")
    if file_sha256(run_dir / "protocol.json") != frozen["protocol_sha256"]:
        raise ValueError("protocol checksum mismatch")
    if any(
        file_sha256(data_dir / name) != provenance["input_sha256"].get(name)
        for name in DATA_FILES
    ):
        raise ValueError("dataset identity mismatch")
    dataset = load_evaluation_dataset(data_dir, "test", verify_all_splits=False)
    ids = read_json(data_dir / "manifest.json")["primary_slice_query_ids"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("unique primary query IDs required")
    if (
        hashlib.sha256("\n".join(ids).encode()).hexdigest()
        != provenance["query_ids_sha256"]
    ):
        raise ValueError("query order checksum mismatch")
    documents = {doc.document_id for doc in dataset.documents}
    records = [
        json.loads(line)
        for line in (run_dir / "per-query.jsonl").read_text().splitlines()
    ]
    grouped = defaultdict(list)
    configs = method_configs(frozen)
    allowed = {name: value["alpha_dense"] for name, value in configs.items()}
    methods = provenance["methods"]
    required = {f"{name} + Fixed" for name in ("baseline", frozen["selected_encoder"])}
    if frozen.get("shared_alpha_dense") is not None:
        required = set(configs)
    if (
        not methods
        or len(set(methods)) != len(methods)
        or set(methods) - allowed.keys()
        or not required <= set(methods)
        or provenance["run_id"] != frozen["run_id"]
    ):
        raise ValueError("invalid method set")
    for row in records:
        method, query_id = row["method"], row["query_id"]
        if (
            method not in methods
            or query_id not in ids
            or row["run_id"] != frozen["run_id"]
        ):
            raise ValueError("unknown method/query/run identity")
        if row["selected_alpha_dense"] != allowed[method]:
            raise ValueError("alpha differs from frozen validation selection")
        if (
            "selection_policy" in row
            and row["selection_policy"] != configs[method]["selection_policy"]
        ):
            raise ValueError("selection policy differs from frozen comparison")
        ranking = row["top10_document_ids"]
        if (
            not ranking
            or len(ranking) > 10
            or len(set(ranking)) != len(ranking)
            or set(ranking) - documents
        ):
            raise ValueError("invalid ranking")
        hits = [SearchResult(doc, rank, 0.0) for rank, doc in enumerate(ranking, 1)]
        labels = dataset.qrels[query_id]
        actual = {
            "p@1": precision_at_k(hits, labels, 1),
            "recall@10": recall_at_k(hits, labels, 10),
            "mrr@10": reciprocal_rank_at_k(hits, labels, 10),
            "ndcg@10": ndcg_at_k(hits, labels, 10),
        }
        if any(
            not isclose(row[key], value, rel_tol=0, abs_tol=1e-12)
            for key, value in actual.items()
        ):
            raise ValueError("saved metrics disagree with rankings and qrels")
        rank = next(
            (i for i, doc in enumerate(ranking, 1) if labels.get(doc, 0) > 0), None
        )
        if row["relevant_rank_top10"] != rank:
            raise ValueError("saved relevant rank differs")
        grouped[method].append(query_id)
    if set(grouped) != set(methods) or any(
        values != ids for values in grouped.values()
    ):
        raise ValueError("missing/duplicate/reordered query or method")
    if len(records) != provenance["rows"] or provenance["queries"] != len(ids):
        raise ValueError("row/query count mismatch")
    aggregates = _averages(records)
    for method, metrics in aggregates.items():
        for key, value in metrics.items():
            if not isclose(
                value, provenance["aggregates"][method][key], rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError("aggregate metrics differ")
    return {
        "verified": True,
        "queries": len(ids),
        "rows": len(records),
        "aggregates": aggregates,
    }


def compact_export(data_dir, run_dir, output_dir, *, canonical_summary=None):
    """Export only baseline/chosen encoder rankings and the JSON needed to verify them."""
    import shutil

    data_dir, run_dir, output_dir = map(Path, (data_dir, run_dir, output_dir))
    verify_results(data_dir, run_dir)
    if output_dir.exists():
        raise ValueError("compact output already exists; use a new directory")
    selection = read_json(run_dir / "frozen.json")
    selected = selection["selected_encoder"]
    methods = {f"{name} + Fixed" for name in ("baseline", selected)}
    if selection.get("shared_alpha_dense") is not None:
        methods = set(method_configs(selection))
    records = [
        json.loads(line)
        for line in (run_dir / "per-query.jsonl").read_text().splitlines()
        if json.loads(line)["method"] in methods
    ]
    reference = None
    if canonical_summary is not None:
        canonical_summary = Path(canonical_summary)
        canonical = read_json(canonical_summary)
        if canonical["dataset"]["manifest_sha256"] != file_sha256(
            data_dir / "manifest.json"
        ):
            raise ValueError("canonical manifest differs from Finance manifest")
        for name in DATA_FILES[1:]:
            if canonical["dataset"]["input_sha256"].get(name) != file_sha256(
                data_dir / name
            ):
                raise ValueError("canonical dataset differs from Finance dataset")
        reference = {
            "role": "saved main canonical reference; not the union-minmax Finance control",
            "source_sha256": file_sha256(canonical_summary),
            "git_commit": canonical["code"]["git_commit"],
            "vector_sha256": canonical["encoding"]["vector_sha256"],
            "metrics": {
                name: {key: value["slices"]["non_synthetic"][key] for key in METRICS}
                for name, value in canonical["methods"].items()
            },
        }
    output_dir.mkdir(parents=True)
    for name in ("frozen.json", "protocol.json"):
        shutil.copyfile(run_dir / name, output_dir / name)
    with (output_dir / "per-query.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    write_manifest(data_dir, output_dir, records)
    manifest_path = output_dir / "per-query-manifest.json"
    manifest = read_json(manifest_path)
    manifest["canonical_reference"] = reference
    manifest["selected_encoder"] = selected
    manifest["evaluation_input_sha256"] = read_json(run_dir / "summary.json")[
        "input_sha256"
    ]
    write_json(manifest_path, manifest)
    return verify_results(data_dir, output_dir)

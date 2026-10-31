"""Extract run identity from existing BC Card reports and refuse mismatched comparisons.

Reports from `run_enterprise_baselines` already carry dataset, encoding, code and
environment blocks. This module only reads them; it does not define a new format.
Protocol fields (data, slice, metric scope) must match before two runs are
compared. Model and fusion fields are what a comparison varies, so they are
listed as differences, and a same-model run with different vectors is flagged.
"""

from pathlib import Path

from evaluation.data import file_sha256
from evaluation.enterprise import _ids_digest as ids_sha256

PRIMARY_SLICE = "non_synthetic"
# 비교 전에 반드시 같아야 하는 항목. 한쪽이라도 없으면 unverified로 남긴다.
PROTOCOL_KEYS = (
    "split",
    "dataset_id",
    "dataset_revision",
    "manifest_sha256",
    "corpus_sha256",
    "queries_sha256",
    "qrels_sha256",
    "corpus_size",
    "query_ids_sha256",
    "primary_slice_query_ids_sha256",
    "top_k",
    "relevance",
)


class ProtocolMismatch(ValueError):
    """Two runs do not share the data, slice or metric protocol."""


def run_identity(report, method, *, manifest=None, report_path=None):
    """Return the identity of one method inside a saved report.

    `manifest` is the prepared dataset manifest; when given, the primary slice
    query IDs are hashed from it so reports written before this module still
    expose the slice they were summarized on.
    """
    if method not in report["methods"]:
        raise KeyError(f"report에 {method} 결과가 없습니다.")
    dataset = report["dataset"]
    split = dataset["split"]
    inputs = dataset.get("input_sha256", {})
    data = report["methods"][method]
    encoding = report.get("encoding") or {}
    model = encoding.get("model") or {}
    identity = {
        "protocol": {
            "split": split,
            "dataset_id": dataset.get("dataset_id"),
            "dataset_revision": dataset.get("dataset_revision"),
            "manifest_sha256": dataset.get("manifest_sha256"),
            "corpus_sha256": inputs.get("corpus.jsonl"),
            "queries_sha256": inputs.get(f"queries_{split}.jsonl"),
            "qrels_sha256": inputs.get(f"qrels_{split}.jsonl"),
            "corpus_size": dataset.get("corpus_size"),
            "query_ids_sha256": ids_sha256(row["query_id"] for row in data["queries"]),
            "top_k": report["conditions"]["top_k"],
            "relevance": report["conditions"]["relevance"],
        },
        "method": {
            "name": method,
            "config": data.get("config"),
            "model_name": model.get("model_name"),
            "model_revision": model.get("revision"),
            "normalize_embeddings": model.get("normalize_embeddings"),
            "query_prefix": model.get("query_prefix"),
            "passage_prefix": model.get("passage_prefix"),
            "dtype": encoding.get("dtype"),
            "max_seq_length": encoding.get("max_seq_length"),
            "embedding_dimension": encoding.get("embedding_dimension"),
            "vector_sha256": encoding.get("vector_sha256"),
        },
        "run": {
            "git_commit": report.get("code", {}).get("git_commit"),
            "working_tree_dirty": report.get("code", {}).get("working_tree_dirty"),
            "created_at_utc": report.get("created_at_utc"),
            "environment": {
                key: report.get("environment", {}).get(key)
                for key in ("python", "platform", "cpu_count", "packages")
            },
        },
    }
    if manifest is not None:
        primary = manifest["primary_slice_query_ids"]
        identity["protocol"]["primary_slice_query_ids_sha256"] = ids_sha256(primary)
        identity["protocol"]["primary_slice_query_count"] = len(primary)
        missing = set(primary) - {row["query_id"] for row in data["queries"]}
        if missing:
            raise ProtocolMismatch(
                f"{method}: primary slice 질의 {len(missing)}개 없음"
            )
    if PRIMARY_SLICE in data.get("slices", {}):
        identity["protocol"]["primary_slice_query_count_in_report"] = data["slices"][
            PRIMARY_SLICE
        ]["query_count"]
    if report_path is not None:
        identity["run"]["result_sha256"] = file_sha256(Path(report_path))
    return identity


def compare_identities(a, b):
    """Raise on protocol differences; return what differs between the two methods.

    Fields missing from either side (older reports) are listed as unverified
    instead of passing silently.
    """
    mismatched, unverified = [], []
    extra = (set(a["protocol"]) | set(b["protocol"])) - set(PROTOCOL_KEYS)
    for key in (*PROTOCOL_KEYS, *sorted(extra)):
        left, right = a["protocol"].get(key), b["protocol"].get(key)
        if left is None or right is None:
            unverified.append(key)
        elif left != right:
            mismatched.append(f"{key}: {left!r} != {right!r}")
    if mismatched:
        raise ProtocolMismatch("protocol 불일치: " + "; ".join(mismatched))
    differences = {
        key: [a["method"][key], b["method"][key]]
        for key in a["method"]
        if a["method"][key] != b["method"].get(key)
    }
    warnings = []
    same_model = (
        a["method"]["model_name"] == b["method"]["model_name"]
        and a["method"]["model_revision"] == b["method"]["model_revision"]
    )
    if same_model and "vector_sha256" in differences:
        warnings.append(
            "같은 model/revision인데 vector SHA-256이 다릅니다. "
            "encoding을 다시 한 run끼리 비교하는 중입니다."
        )
    if a["run"]["working_tree_dirty"] or b["run"]["working_tree_dirty"]:
        warnings.append("dirty working tree에서 만든 run이 있습니다.")
    return {"differences": differences, "unverified": unverified, "warnings": warnings}

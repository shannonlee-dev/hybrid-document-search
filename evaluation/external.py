"""Normalize, checkpoint and evaluate external (commercial) retrieval runs.

Provider hits are mapped to BC Card `source_chunk_id`s through an explicit
ID map, deduplicated keeping the best rank, and scored with the shared metric
functions on the frozen #25 qrels. Failed or missing queries stay in every
denominator with empty results and are counted separately.
"""

import hashlib
import json
import os
from datetime import UTC, datetime
from math import fsum
from pathlib import Path
from time import perf_counter, sleep

from evaluation.data import file_sha256, load_evaluation_dataset
from evaluation.enterprise import (
    REPORT_METRICS,
    query_metrics,
    slice_definitions,
    summarize,
)
from evaluation.json_io import read_json, write_json
from evaluation.latency import latency_summary
from evaluation.metrics import evaluate_query
from retrievers.base import SearchResult

RUN_SCHEMA_VERSION = "1"
# #25가 남긴 업로드 차단 상태. 사람이 검토한 기록 없이 외부 전송 명령을 실행하지 않는다.
UPLOAD_GATE = "EXTERNAL_UPLOAD_REQUIRES_MANUAL_TERMS_CHECK"
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0
RUN_FILE, RECORDS_FILE, RAW_FILE = "run.json", "run.jsonl", "raw-responses.jsonl"
INDEX_IDENTITY_SUFFIX = ".identity.json"
LOCAL_REPORT_SLICES = ("non_synthetic", "all")
_RUN_IDENTITY = (
    "provider",
    "provider_settings",
    "split",
    "top_k",
    "dataset_manifest_sha256",
    "query_set_sha256",
    "id_map_sha256",
)
_INDEX_IDENTITY = (
    "provider",
    "provider_settings",
    "dataset_manifest_sha256",
    "corpus_sha256",
    "document_set_sha256",
)


class NormalizationError(ValueError):
    def __init__(self, message, kind):
        super().__init__(message)
        self.kind = kind


class RunAborted(RuntimeError):
    pass


def _digest(lines):
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def identity_id_map(documents):
    """Provider document ID = BC Card source_chunk_id (one chunk per indexed unit)."""
    return {document.document_id: document.document_id for document in documents}


def id_map_digest(id_map):
    return _digest(f"{key}\t{value}" for key, value in sorted(id_map.items()))


def query_set_digest(queries):
    return _digest(f"{query.query_id}\t{query.text}" for query in queries)


def document_set_digest(documents):
    """Digest of the actual indexed document set (ID + text), order-independent.

    A full-corpus hash would miss a `--limit` selection change that keeps the
    same corpus but indexes a different subset, so this hashes the documents
    actually passed to `index_documents`, not the corpus they came from.
    """
    ordered = sorted(documents, key=lambda document: document.document_id)
    return _digest(f"{document.document_id}\t{document.text}" for document in ordered)


def _index_identity_path(checkpoint):
    return Path(checkpoint).with_name(Path(checkpoint).name + INDEX_IDENTITY_SUFFIX)


def require_terms_review(path, provider):
    """Load the manual terms decision; refuse anything but an explicit approval."""
    if path is None:
        raise ValueError(
            f"{UPLOAD_GATE}: --terms-review에 약관 검토 기록(JSON)이 필요합니다."
        )
    record = read_json(path)
    required = ("provider", "decision", "reviewer", "reviewed_at", "evidence")
    if any(not record.get(key) for key in required):
        raise ValueError(f"약관 검토 기록에 {', '.join(required)}가 필요합니다.")
    if record["provider"] != provider or record["decision"] != "approved":
        raise ValueError(f"{UPLOAD_GATE}: {provider} 업로드가 승인되지 않았습니다.")
    return record


def normalize_hits(hits, id_map, top_k):
    """Map provider hits to original IDs; keep each original's best rank.

    Provider order is used unless every hit has a rank. A hit whose provider ID
    is missing or unknown, or whose metadata disagrees with the map, fails the
    query instead of being dropped, so rankings are never silently shifted.
    """
    hits = list(hits)
    if hits and all(hit.rank is not None for hit in hits):
        hits.sort(key=lambda hit: hit.rank)
    rows, seen, duplicates = [], set(), 0
    for hit in hits:
        if not isinstance(hit.provider_id, str) or not hit.provider_id.strip():
            raise NormalizationError(
                "provider 결과 ID가 없습니다.", "missing_result_id"
            )
        original = id_map.get(hit.provider_id)
        if original is None:
            raise NormalizationError(
                f"ID map에 없는 provider ID: {hit.provider_id}", "unmapped_id"
            )
        if hit.original_id is not None and hit.original_id != original:
            raise NormalizationError(
                f"metadata 원본 ID 불일치: {hit.provider_id}", "conflicting_original_id"
            )
        # 한 원본 chunk가 여러 provider sub-chunk로 나뉘면 가장 높은 순위 하나만 남긴다.
        if original in seen:
            duplicates += 1
            continue
        seen.add(original)
        rows.append(
            {
                "rank": len(rows) + 1,
                "original_document_id": original,
                "provider_document_id": hit.provider_id,
                "provider_chunk": hit.chunk,
                "provider_rank": hit.rank,
                "score": hit.score,
            }
        )
    kept = rows[:top_k]
    return kept, {
        "provider_hit_count": len(hits),
        "duplicate_hits": duplicates,
        "unique_count": len(kept),
        "short_of_top_k": len(kept) < top_k,
        "rank_source": "provider rank"
        if hits and hits[0].rank is not None
        else "order",
    }


def _append(path, record):
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        target.flush()
        os.fsync(target.fileno())


def _trim_partial_line(path):
    """Drop an unterminated last line so resumed appends start on a new line."""
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        with path.open("r+b") as target:
            target.truncate(data.rfind(b"\n") + 1)


def read_jsonl(path):
    """Read records; a truncated last line from an interrupted write is ignored."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if number == len(lines):
                break
            raise ValueError(f"{path.name}:{number}: JSON 형식이 올바르지 않습니다.")
    return records


def latest_records(path):
    return {record["query_id"]: record for record in read_jsonl(path)}


def _with_retries(call, *, max_attempts, backoff, sleeper, clock):
    """Run call() with bounded exponential backoff for transient errors only."""
    from commercial.base import ProviderError

    attempts, waited, timings = 0, 0.0, []
    while True:
        attempts += 1
        started = clock()
        try:
            value = call()
        except ProviderError as exc:
            timings.append((clock() - started) * 1000)
            if exc.abort or not exc.transient or attempts >= max_attempts:
                exc.attempts, exc.waited, exc.timings = attempts, waited, timings
                raise
            delay = min(MAX_BACKOFF_SECONDS, backoff * 2 ** (attempts - 1))
            if exc.retry_after is not None:
                delay = min(MAX_BACKOFF_SECONDS, max(delay, exc.retry_after))
            if exc.kind == "rate_limited":
                waited += delay
            sleeper(delay)
            continue
        timings.append((clock() - started) * 1000)
        return value, attempts, waited, timings


def run_metadata(provider, dataset, id_map, top_k, *, terms_review=None):
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "provider": provider.name,
        "provider_settings": provider.describe(),
        "split": dataset.split,
        "top_k": top_k,
        "dataset_manifest_sha256": dataset.provenance["manifest_sha256"],
        "corpus_sha256": dataset.provenance["input_sha256"]["corpus.jsonl"],
        "query_set_sha256": query_set_digest(dataset.queries),
        "id_map_sha256": id_map_digest(id_map),
        "id_mapping": "provider docId = source_chunk_id; extra.source_chunk_id copy",
        "normalization": (
            "sort by provider rank when every hit has one, else provider order; "
            "map via ID map; drop repeated originals keeping best rank; truncate "
            "to top_k; no padding; unknown or missing IDs fail the query"
        ),
        "concurrency": 1,
        "terms_review": terms_review,
    }


def run_external_queries(
    provider,
    queries,
    output_dir,
    *,
    metadata,
    id_map,
    max_attempts=DEFAULT_MAX_ATTEMPTS,
    backoff=DEFAULT_BACKOFF_SECONDS,
    keep_raw=True,
    sleeper=sleep,
    clock=perf_counter,
    progress=None,
):
    """Query sequentially with a per-query JSONL checkpoint; resume skips successes.

    Failed queries are recorded and retried on the next resume. An abort error
    (auth, missing index, quota) stops the run without recording that query.
    """
    from commercial.base import ProviderError

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / RUN_FILE
    if run_path.exists():
        saved = read_json(run_path)
        changed = [key for key in _RUN_IDENTITY if saved.get(key) != metadata[key]]
        if changed:
            raise ValueError(f"기존 checkpoint와 실행 조건이 다릅니다: {changed}")
        metadata = saved
    else:
        metadata = {**metadata, "started_at_utc": datetime.now(UTC).isoformat()}
        write_json(run_path, metadata)
    _trim_partial_line(output_dir / RECORDS_FILE)
    _trim_partial_line(output_dir / RAW_FILE)
    done = {
        q
        for q, record in latest_records(output_dir / RECORDS_FILE).items()
        if record["status"] == "ok"
    }
    top_k = metadata["top_k"]
    pending = [query for query in queries if query.query_id not in done]
    for position, query in enumerate(pending, start=1):
        record = {"query_id": query.query_id}
        try:
            response, attempts, waited, timings = _with_retries(
                lambda q=query: provider.search(q.text, top_k),
                max_attempts=max_attempts,
                backoff=backoff,
                sleeper=sleeper,
                clock=clock,
            )
            if keep_raw:
                _append(
                    output_dir / RAW_FILE,
                    {"query_id": query.query_id, "response": response.raw},
                )
            record |= {
                "attempts": attempts,
                "rate_limit_wait_seconds": waited,
                "latency_ms": timings[-1],
                "attempt_latency_ms": timings,
                "usage": response.usage,
            }
            results, diagnostics = normalize_hits(response.hits, id_map, top_k)
            record |= {"status": "ok", "error": None, "results": results, **diagnostics}
        except ProviderError as exc:
            if exc.abort:
                raise RunAborted(f"{exc.kind}: {exc}") from exc
            record |= {
                "status": "failed",
                "error": {"kind": exc.kind, "message": str(exc)},
                "attempts": exc.attempts,
                "rate_limit_wait_seconds": exc.waited,
                "attempt_latency_ms": exc.timings,
                "results": [],
            }
        except NormalizationError as exc:
            record |= {
                "status": "failed",
                "error": {"kind": exc.kind, "message": str(exc)},
                "results": [],
            }
        record["completed_at_utc"] = datetime.now(UTC).isoformat()
        _append(output_dir / RECORDS_FILE, record)
        if progress and (position % 50 == 0 or position == len(pending)):
            progress(f"{position}/{len(pending)} queries")
    return latest_records(output_dir / RECORDS_FILE)


def index_documents(
    client,
    documents,
    checkpoint,
    *,
    run_identity,
    max_attempts=DEFAULT_MAX_ATTEMPTS,
    backoff=DEFAULT_BACKOFF_SECONDS,
    sleeper=sleep,
    clock=perf_counter,
    progress=None,
):
    """Upload documents one by one under their own IDs, resuming from checkpoint.

    `run_identity` must carry provider/provider_settings and corpus identity
    (no credentials); the actual selected document set is hashed here. Identity
    is written before the first upload and checked before any resume, so a
    changed provider, service, corpus or `--limit` selection is refused before
    `client.add_document()` is ever called.
    """
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    identity_path = _index_identity_path(checkpoint)
    identity = {**run_identity, "document_set_sha256": document_set_digest(documents)}
    if identity_path.exists():
        saved = read_json(identity_path)
        changed = [key for key in _INDEX_IDENTITY if saved.get(key) != identity[key]]
        if changed:
            raise ValueError(f"기존 색인 checkpoint와 실행 조건이 다릅니다: {changed}")
    else:
        write_json(identity_path, identity)
    _trim_partial_line(checkpoint)
    done = {row["document_id"] for row in read_jsonl(checkpoint)}
    pending = [doc for doc in documents if doc.document_id not in done]
    started = clock()
    for position, document in enumerate(pending, start=1):
        _, attempts, _, timings = _with_retries(
            lambda d=document: client.add_document(d.document_id, d.text),
            max_attempts=max_attempts,
            backoff=backoff,
            sleeper=sleeper,
            clock=clock,
        )
        _append(
            checkpoint,
            {
                "document_id": document.document_id,
                "attempts": attempts,
                "latency_ms": timings[-1],
                "uploaded_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        if progress and (position % 500 == 0 or position == len(pending)):
            progress(f"{position}/{len(pending)} documents")
    return {
        "uploaded_now": len(pending),
        "already_uploaded": len(done),
        "upload_seconds": clock() - started,
    }


def evaluate_external_run(run_dir, data_dir, *, local_summary=None):
    """Score a saved run on its split with failures counted as empty rankings."""
    run_dir = Path(run_dir)
    metadata = read_json(run_dir / RUN_FILE)
    dataset = load_evaluation_dataset(data_dir, metadata["split"])
    if metadata["dataset_manifest_sha256"] != dataset.provenance["manifest_sha256"]:
        raise ValueError("run이 다른 dataset manifest로 생성되었습니다.")
    if metadata["query_set_sha256"] != query_set_digest(dataset.queries):
        raise ValueError("run의 query set이 현재 split과 다릅니다.")
    corpus_ids = {document.document_id for document in dataset.documents}
    records = latest_records(run_dir / RECORDS_FILE)
    unknown_queries = set(records) - set(dataset.qrels)
    if unknown_queries:
        raise ValueError(f"split에 없는 query_id: {sorted(unknown_queries)[:3]}")

    from data.bccard import load_query_metadata

    slices = slice_definitions(load_query_metadata(data_dir, dataset.split))
    per_query, status = {}, {}
    for query in dataset.queries:
        record = records.get(query.query_id)
        status[query.query_id] = "missing" if record is None else record["status"]
        rows = record["results"] if record and record["status"] == "ok" else []
        unknown = {row["original_document_id"] for row in rows} - corpus_ids
        if unknown:
            raise ValueError(f"corpus에 없는 원본 ID: {sorted(unknown)[:3]}")
        # 지표 함수는 score를 쓰지 않는다. score를 노출하지 않는 provider는 0.0으로 둔다.
        hits = [
            SearchResult(row["original_document_id"], row["rank"], row["score"] or 0.0)
            for row in rows
        ]
        judgments = dataset.qrels[query.query_id]
        metrics = evaluate_query(hits, judgments)
        metrics |= query_metrics([vars(hit) for hit in hits], judgments)
        metrics["mrr@10"] = metrics["rr@10"]
        per_query[query.query_id] = metrics

    def counts(ids):
        short = sum(
            1 for q in ids if status[q] == "ok" and records[q]["short_of_top_k"]
        )
        return {
            "failed": sum(status[q] == "failed" for q in ids),
            "missing": sum(status[q] == "missing" for q in ids),
            "short_of_top_k": short,
        }

    ok = [records[q] for q in status if status[q] == "ok"]
    failures = {}
    for q, value in status.items():
        if value == "failed":
            kind = records[q]["error"]["kind"]
            failures[kind] = failures.get(kind, 0) + 1
    usage = {}
    for record in ok:
        for key, value in (record.get("usage") or {}).items():
            if isinstance(value, int):
                usage[key] = usage.get(key, 0) + value
    report = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run": metadata,
        "dataset": {
            "split": dataset.split,
            "manifest_sha256": dataset.provenance["manifest_sha256"],
            "query_count": len(dataset.queries),
        },
        "code_sha256": {
            "evaluation/external.py": file_sha256(Path(__file__)),
        },
        "queries": {
            "requested": len(dataset.queries),
            "ok": len(ok),
            "failed": sum(v == "failed" for v in status.values()),
            "missing": sum(v == "missing" for v in status.values()),
            "failure_kinds": failures,
            "duplicate_hits_removed": sum(r["duplicate_hits"] for r in ok),
            "short_of_top_k": sum(r["short_of_top_k"] for r in ok),
            "missing_score_hits": sum(
                row["score"] is None for r in ok for row in r["results"]
            ),
        },
        "latency": {
            "scope": "end-to-end HTTPS request incl. provider query rewrite, "
            "retrieval and generation; final attempt per query; not comparable "
            "to local cached-vector CPU latency",
            "concurrency": metadata.get("concurrency", 1),
            "request": latency_summary([r["latency_ms"] for r in ok]) if ok else None,
            "retries": sum(r["attempts"] - 1 for r in ok),
            "rate_limit_wait_seconds": fsum(r["rate_limit_wait_seconds"] for r in ok),
        },
        "usage_totals": usage,
        "cost": "not computed; read billed amounts from the provider console",
        "slices": {
            name: summarize(per_query, ids) | counts(ids)
            for name, ids in slices.items()
            if ids
        },
    }
    if local_summary is not None:
        report["local_baselines"] = local_baseline_rows(local_summary, dataset.split)
    return report


def local_baseline_rows(summary, split):
    """Copy published #25 baseline metrics; local latency keeps its own scope."""
    if summary["dataset"]["split"] != split:
        raise ValueError("local baseline summary의 split이 run과 다릅니다.")
    return {
        "latency_scope": summary["conditions"]["latency_scope"],
        "methods": {
            name: {
                "slices": {
                    s: {
                        k: data["slices"][s][k]
                        for k in ("query_count", *REPORT_METRICS)
                    }
                    for s in LOCAL_REPORT_SLICES
                },
                "cpu_latency_ms": [
                    data["latency"]["mean_ms"],
                    data["latency"]["p95_ms"],
                ],
            }
            for name, data in summary["methods"].items()
        },
    }

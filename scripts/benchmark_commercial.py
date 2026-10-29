"""Index BC Card into a commercial RAG service, collect Top-K, and evaluate offline.

Every command that sends data to the provider requires a manual terms review
record (EXTERNAL_UPLOAD_REQUIRES_MANUAL_TERMS_CHECK) and credentials from the
environment. `evaluate` reads saved runs only and needs neither.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, sleep

from evaluation.constants import EVALUATION_TOP_K
from scripts._cli import CliArgumentParser, _positive_int

PROVIDERS = ("naver",)
DEFAULT_LOCAL_SUMMARY = Path("docs/experiments/bccard/test-summary.json")


def main(argv=None):
    parser = CliArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("index", "smoke", "search", "evaluate"))
    parser.add_argument("--provider", choices=PROVIDERS, default="naver")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/bccard"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--terms-review", type=Path)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--documents", type=_positive_int, default=3)
    parser.add_argument("--top-k", type=_positive_int, default=EVALUATION_TOP_K)
    parser.add_argument("--index-timeout", type=_positive_int, default=3600)
    parser.add_argument("--local-summary", type=Path, default=DEFAULT_LOCAL_SUMMARY)
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            if args.run_dir is None or args.output_dir is None:
                parser.error("evaluate에는 --run-dir과 --output-dir이 필요합니다.")
            _evaluate(args)
            return 0
        if args.output_dir is None:
            parser.error(f"{args.command}에는 --output-dir이 필요합니다.")
        {"index": _index, "smoke": _smoke, "search": _search}[args.command](args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(f"상용 서비스 benchmark 실패: {exc}")
    return 0


def _progress(message):
    print(message, file=sys.stderr, flush=True)


def _client(args):
    from commercial.naver import NaverRagClient

    return NaverRagClient.from_env()


def _gate(args):
    from commercial.naver import NaverRagClient
    from evaluation.external import require_terms_review

    return require_terms_review(args.terms_review, NaverRagClient.name)


def _wait_for_count(client, expected, timeout):
    started = perf_counter()
    while True:
        count = client.document_count()
        if count is not None and count >= expected:
            return count, perf_counter() - started
        if perf_counter() - started > timeout:
            raise RuntimeError(f"색인 대기 시간 초과: {count}/{expected}")
        sleep(10)


def _index(args):
    from evaluation.data import load_evaluation_dataset
    from evaluation.external import id_map_digest, identity_id_map, index_documents
    from evaluation.json_io import write_json

    review = _gate(args)
    dataset = load_evaluation_dataset(args.data_dir, args.split)
    documents = dataset.documents[: args.limit] if args.limit else dataset.documents
    client = _client(args)
    run_identity = {
        "provider": client.name,
        "provider_settings": client.describe(),
        "dataset_manifest_sha256": dataset.provenance["manifest_sha256"],
        "corpus_sha256": dataset.provenance["input_sha256"]["corpus.jsonl"],
    }
    started_at = datetime.now(UTC).isoformat()
    upload = index_documents(
        client,
        documents,
        args.output_dir / "indexed.jsonl",
        run_identity=run_identity,
        progress=_progress,
    )
    count, wait_seconds = _wait_for_count(client, len(documents), args.index_timeout)
    record = {
        "provider_settings": client.describe(),
        "started_at_utc": started_at,
        "document_count_requested": len(documents),
        "provider_document_count": count,
        "corpus_sha256": dataset.provenance["input_sha256"]["corpus.jsonl"],
        "id_map_sha256": id_map_digest(identity_id_map(dataset.documents)),
        "indexed_text": "Pair positive text only; title not indexed; unchanged",
        **upload,
        "count_wait_seconds": wait_seconds,
        "terms_review": review,
    }
    write_json(args.output_dir / "index.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


def _smoke(args):
    """Round-trip IDs on a few validation documents; checks mapping, not quality."""
    from evaluation.data import load_evaluation_dataset
    from evaluation.external import identity_id_map, index_documents, normalize_hits

    _gate(args)
    dataset = load_evaluation_dataset(args.data_dir, "validation")
    id_map = identity_id_map(dataset.documents)
    texts = {document.document_id: document for document in dataset.documents}
    queries = dataset.queries[: args.documents]
    targets = [
        texts[max(dataset.qrels[q.query_id], key=dataset.qrels[q.query_id].get)]
        for q in queries
    ]
    client = _client(args)
    run_identity = {
        "provider": client.name,
        "provider_settings": client.describe(),
        "dataset_manifest_sha256": dataset.provenance["manifest_sha256"],
        "corpus_sha256": dataset.provenance["input_sha256"]["corpus.jsonl"],
    }
    index_documents(
        client,
        targets,
        args.output_dir / "smoke-indexed.jsonl",
        run_identity=run_identity,
    )
    deadline = perf_counter() + args.index_timeout
    while any(client.document_status(d.document_id) != "DONE" for d in targets):
        if perf_counter() > deadline:
            raise RuntimeError("smoke 문서 색인이 끝나지 않았습니다.")
        sleep(5)
    report = []
    for document in targets:
        response = client.search(document.text, args.top_k)
        rows, diagnostics = normalize_hits(response.hits, id_map, args.top_k)
        report.append(
            {
                "document_id": document.document_id,
                "returned": [row["original_document_id"] for row in rows],
                "self_retrieved": any(
                    row["original_document_id"] == document.document_id for row in rows
                ),
                **diagnostics,
            }
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(row["self_retrieved"] for row in report):
        raise RuntimeError("문서 본문으로 검색해도 같은 ID가 돌아오지 않았습니다.")


def _search(args):
    from evaluation.data import load_evaluation_dataset
    from evaluation.external import (
        identity_id_map,
        run_external_queries,
        run_metadata,
    )

    review = _gate(args)
    dataset = load_evaluation_dataset(args.data_dir, args.split)
    client = _client(args)
    id_map = identity_id_map(dataset.documents)
    queries = dataset.queries[: args.limit] if args.limit else dataset.queries
    metadata = run_metadata(client, dataset, id_map, args.top_k, terms_review=review)
    records = run_external_queries(
        client,
        queries,
        args.output_dir,
        metadata=metadata,
        id_map=id_map,
        progress=_progress,
    )
    statuses = [record["status"] for record in records.values()]
    print(
        json.dumps(
            {
                "requested": len(queries),
                "ok": statuses.count("ok"),
                "failed": statuses.count("failed"),
            }
        )
    )


def _evaluate(args):
    from evaluation.external import evaluate_external_run
    from evaluation.json_io import read_json, write_json

    summary = read_json(args.local_summary) if args.local_summary.exists() else None
    report = evaluate_external_run(args.run_dir, args.data_dir, local_summary=summary)
    write_json(args.output_dir / "summary.json", report)
    printed = {
        "queries": report["queries"],
        "latency": report["latency"]["request"],
        **{name: report["slices"][name] for name in ("non_synthetic", "all")},
    }
    print(json.dumps(printed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

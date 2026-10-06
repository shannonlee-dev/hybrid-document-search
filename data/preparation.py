"""Ko-miracl의 재현 가능한 공통 subset을 준비합니다."""

import hashlib
import heapq
import json
import math
import random
import re
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

DATASET_ID = "taeminlee/Ko-miracl"
PINNED_REVISION = "5c7690518e481375551916f24241048cf7b017d0"
SOURCE_FILES = ("corpus.jsonl", "queries.jsonl", "qrels/train.jsonl", "qrels/dev.jsonl")


def file_sha256(path: Path) -> str:
    """파일을 스트리밍하여 체크섬을 계산합니다."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_revision(revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision은 고정된 40자리 commit SHA여야 합니다.")


def _positive(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name}는 양의 정수여야 합니다.")


def _string(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}는 비어 있지 않은 문자열이어야 합니다.")
    return value


def _query_id(value) -> str:
    # 실제 원본 qrels는 정수 query-id를 사용합니다.
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return _string(value, "query_id")


def _records(path: Path):
    with path.open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("JSON 객체여야 합니다.")
                yield record
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path.name}:{number}: {exc}") from exc


def _qrels(path: Path, queries: dict) -> tuple[list[dict], int]:
    judgments = {}
    raw_count = 0
    for record in _records(path):
        raw_count += 1
        query_id = _query_id(record.get("query-id"))
        document_id = _string(record.get("corpus-id"), "document_id")
        relevance = record.get("score")
        if (
            isinstance(relevance, bool)
            or not isinstance(relevance, (int, float))
            or (isinstance(relevance, float) and not math.isfinite(relevance))
        ):
            raise ValueError("relevance는 유한한 숫자여야 합니다.")
        if query_id not in queries:
            raise ValueError(f"존재하지 않는 query_id: {query_id}")
        pair = (query_id, document_id)
        if pair in judgments and judgments[pair] != relevance:
            raise ValueError(f"충돌하는 중복 qrel: {pair}")
        judgments[pair] = relevance

    rows = [
        {"query_id": query_id, "document_id": document_id, "relevance": relevance}
        for (query_id, document_id), relevance in sorted(judgments.items())
    ]
    return rows, raw_count


def _write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def download_sources(raw_root: str | Path, revision: str = PINNED_REVISION) -> Path:
    """고정 revision을 별도 디렉터리에 다운로드하며 기존 파일을 재사용합니다."""
    _check_revision(revision)
    directory = Path(raw_root) / revision
    directory.mkdir(parents=True, exist_ok=True)
    for name in (*SOURCE_FILES, "README.md"):
        destination = directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            continue
        url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{revision}/{name}"
        print(f"다운로드: {name}", flush=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, delete=False
            ) as target:
                temporary = Path(target.name)
                with urlopen(url, timeout=60) as response:
                    shutil.copyfileobj(response, target)
            temporary.replace(destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return directory


def prepare_dataset(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    corpus_size: int = 10000,
    train_queries: int = 100,
    dev_queries: int = 50,
    seed: int = 42,
    revision: str = PINNED_REVISION,
) -> dict:
    """원본 JSONL을 검증하고 준비 결과와 manifest를 생성합니다."""
    _check_revision(revision)
    for name, value in (
        ("corpus_size", corpus_size),
        ("train_queries", train_queries),
        ("dev_queries", dev_queries),
    ):
        _positive(value, name)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed는 정수여야 합니다.")

    raw = Path(raw_dir)
    output = Path(output_dir)
    sources = {name: raw / name for name in SOURCE_FILES}
    for path in sources.values():
        if not path.is_file():
            raise ValueError(f"원본 파일이 없습니다: {path}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output_dir은 비어 있는 디렉터리여야 합니다.")
    if output.resolve() == raw.resolve() or output.resolve() in raw.resolve().parents:
        raise ValueError("원본 디렉터리를 출력 경로로 사용할 수 없습니다.")

    queries = {}
    for row in _records(sources["queries.jsonl"]):
        query_id = _query_id(row.get("_id"))
        text = _string(row.get("text"), "query text")
        if query_id in queries:
            raise ValueError(f"중복 query_id: {query_id}")
        queries[query_id] = {"query_id": query_id, "text": text}

    judgments = {}
    raw_qrel_counts = {}
    selected_queries = {}
    selected_qrels = {}
    for split, size in (("train", train_queries), ("dev", dev_queries)):
        judgments[split], raw_qrel_counts[split] = _qrels(
            sources[f"qrels/{split}.jsonl"], queries
        )
        eligible = sorted({row["query_id"] for row in judgments[split]})
        if len(eligible) < size:
            raise ValueError(
                f"{split} 질의가 부족합니다: 필요 {size}, 실제 {len(eligible)}"
            )
        selected = sorted(random.Random(seed).sample(eligible, size))
        selected_ids = set(selected)
        selected_queries[split] = selected
        selected_qrels[split] = [
            row for row in judgments[split] if row["query_id"] in selected_ids
        ]

    train_ids = {row["query_id"] for row in judgments["train"]}
    dev_ids = {row["query_id"] for row in judgments["dev"]}
    if train_ids & dev_ids:
        raise ValueError("동일 query_id가 train/dev에 함께 등장합니다.")

    required_ids = {
        row["document_id"] for rows in selected_qrels.values() for row in rows
    }
    all_judged_ids = {row["document_id"] for rows in judgments.values() for row in rows}
    effective_size = max(corpus_size, len(required_ids))
    background_size = effective_size - len(required_ids)
    required_documents = {}
    seen_ids = set()

    def background_documents():
        for row in _records(sources["corpus.jsonl"]):
            document_id = _string(row.get("_id"), "document_id")
            text = _string(row.get("text"), "document text")
            title = row.get("title")
            if title is not None and not isinstance(title, str):
                raise ValueError("title은 문자열 또는 null이어야 합니다.")
            if isinstance(title, str) and not title.strip():
                title = None
            if document_id in seen_ids:
                raise ValueError(f"중복 document_id: {document_id}")
            seen_ids.add(document_id)
            document = {"document_id": document_id, "text": text, "title": title}
            if document_id in required_ids:
                required_documents[document_id] = document
            else:
                yield document

    candidates = background_documents()
    if background_size:
        background = heapq.nsmallest(
            background_size,
            candidates,
            key=lambda document: (
                hashlib.sha256(
                    f"{seed}\0{document['document_id']}".encode("utf-8")
                ).digest(),
                document["document_id"],
            ),
        )
    else:
        background = []
        # nsmallest(0, ...)는 iterator를 읽지 않으므로 직접 검증합니다.
        for _ in candidates:
            pass

    missing = all_judged_ids - seen_ids
    if missing:
        raise ValueError(f"corpus에 없는 qrel document_id: {sorted(missing)[:5]}")
    if len(background) < background_size:
        raise ValueError("목표 규모를 구성할 corpus 문서가 부족합니다.")
    documents = sorted(
        [*required_documents.values(), *background],
        key=lambda document: document["document_id"],
    )

    payloads = {"corpus.jsonl": documents}
    for split in ("train", "dev"):
        payloads[f"queries_{split}.jsonl"] = [
            queries[query_id] for query_id in selected_queries[split]
        ]
        payloads[f"qrels_{split}.jsonl"] = selected_qrels[split]

    manifest = {
        "schema_version": "1",
        "dataset_id": DATASET_ID,
        "dataset_revision": revision,
        "input_mode": "local_jsonl",
        "usage_terms": {
            "status": "conversion_card_has_no_explicit_license",
            "dataset_card": f"https://huggingface.co/datasets/{DATASET_ID}/blob/{revision}/README.md",
            "upstream": "https://github.com/project-miracl/miracl",
            "note": "변환 데이터셋의 별도 이용 조건 확인이 필요합니다.",
        },
        "settings": {
            "corpus_size": corpus_size,
            "train_queries": train_queries,
            "dev_queries": dev_queries,
            "seed": seed,
        },
        "selection": {
            "queries": "split별 정렬한 ID 목록에서 random.Random(seed).sample",
            "background": "SHA256(seed + NUL + document_id) 우선순위의 최소값",
            "effective_corpus_size": effective_size,
            "required_documents": len(required_ids),
            "background_documents": len(background),
            "corpus_size_adjusted": effective_size != corpus_size,
            "adjustment_reason": (
                "선택한 질의의 모든 판정 문서를 포함하기 위해 목표 규모를 늘렸습니다."
                if effective_size != corpus_size
                else None
            ),
        },
        "source_counts": {
            "corpus": len(seen_ids),
            "queries": len(queries),
            "qrels_train": raw_qrel_counts["train"],
            "qrels_dev": raw_qrel_counts["dev"],
            "eligible_train_queries": len(train_ids),
            "eligible_dev_queries": len(dev_ids),
        },
        "output_counts": {name: len(rows) for name, rows in payloads.items()},
        "selected_query_ids": selected_queries,
        "selected_document_ids": [row["document_id"] for row in documents],
        "label_values": {
            split: sorted({row["relevance"] for row in judgments[split]})
            for split in ("train", "dev")
        },
        "source_sha256": {name: file_sha256(path) for name, path in sources.items()},
        "script_sha256": file_sha256(Path(__file__)),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        stage = Path(staging)
        for name, rows in payloads.items():
            _write_jsonl(stage / name, rows)
        manifest["output_sha256"] = {
            name: file_sha256(stage / name) for name in payloads
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        output.mkdir(exist_ok=True)
        for path in stage.iterdir():
            path.replace(output / path.name)
    return manifest

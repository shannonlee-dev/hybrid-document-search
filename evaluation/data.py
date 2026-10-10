"""Validate prepared documents, train/dev queries, qrels and their manifest."""

import hashlib
import json
import re
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from data.loader import Document, load_prepared_documents
from data.preparation import PINNED_REVISION
from evaluation.constants import DATASET_SETTINGS
from evaluation.json_io import read_json


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


@dataclass(frozen=True)
class EvaluationDataset:
    documents: tuple[Document, ...]
    queries: tuple[Query, ...]
    qrels: dict[str, dict[str, float]]
    split: str
    provenance: dict


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_evaluation_dataset(
    directory: str | Path, split: str = "dev"
) -> EvaluationDataset:
    """Load and validate the manifest's prepared files without resampling."""
    if split not in {"train", "dev"}:
        raise ValueError("split은 train 또는 dev여야 합니다.")
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1":
        raise ValueError("지원하지 않는 manifest schema_version입니다.")
    revision = manifest.get("dataset_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("manifest dataset_revision은 고정된 40자리 SHA여야 합니다.")
    if not isinstance(manifest.get("output_sha256"), dict):
        raise ValueError("manifest output_sha256가 필요합니다.")
    if not isinstance(manifest.get("output_counts"), dict):
        raise ValueError("manifest output_counts가 필요합니다.")

    if not isinstance(manifest.get("selected_query_ids", {}), dict):
        raise ValueError("manifest selected_query_ids는 split별 mapping이어야 합니다.")

    # 한 split만 평가해도 전체 준비 데이터가 같은 manifest에 속하는지 확인한다.
    names = ["corpus.jsonl"]
    for part in ("train", "dev"):
        for kind in ("queries", "qrels"):
            names.append(f"{kind}_{part}.jsonl")
    hashes = {}
    for name in names:
        hashes[name] = file_sha256(directory / name)
        if hashes[name] != manifest["output_sha256"].get(name):
            raise ValueError(f"manifest 체크섬과 파일이 다릅니다: {name}")

    documents = tuple(load_prepared_documents(directory / "corpus.jsonl"))
    if not documents:
        raise ValueError("평가 corpus가 비어 있습니다.")
    _check_count(manifest, "corpus.jsonl", len(documents))
    selected_documents = manifest.get("selected_document_ids")
    if selected_documents is not None and selected_documents != [
        d.document_id for d in documents
    ]:
        raise ValueError("manifest selected_document_ids와 corpus가 다릅니다.")
    corpus_ids = {document.document_id for document in documents}
    all_queries = {}
    all_qrels = {}
    for part in ("train", "dev"):
        query_path = directory / f"queries_{part}.jsonl"
        queries = load_queries(query_path)
        _check_count(manifest, query_path.name, len(queries))
        selected_ids = manifest.get("selected_query_ids", {}).get(part)
        if selected_ids is not None and selected_ids != [
            query.query_id for query in queries
        ]:
            raise ValueError(f"manifest selected_query_ids와 {part} 질의가 다릅니다.")
        qrel_path = directory / f"qrels_{part}.jsonl"
        judgments, row_count = load_qrels(qrel_path, queries, corpus_ids)
        _check_count(manifest, qrel_path.name, row_count)
        all_queries[part] = queries
        all_qrels[part] = judgments
    if {query.query_id for query in all_queries["train"]} & {
        query.query_id for query in all_queries["dev"]
    }:
        raise ValueError("train과 dev에 같은 query_id가 있습니다.")

    provenance = {
        "dataset_id": manifest.get("dataset_id"),
        "dataset_revision": revision,
        "manifest_sha256": file_sha256(manifest_path),
        "settings": manifest.get("settings", {}),
        "input_sha256": hashes,
        "corpus_size": len(documents),
        "query_count": len(all_queries[split]),
        "qrel_count": manifest["output_counts"][f"qrels_{split}.jsonl"],
        "query_ids": [query.query_id for query in all_queries[split]],
    }
    return EvaluationDataset(
        documents, all_queries[split], all_qrels[split], split, provenance
    )


def load_queries(path: str | Path) -> tuple[Query, ...]:
    queries = []
    seen = set()
    for location, row in _records(Path(path)):
        query_id = _string(row.get("query_id"), "query_id", location)
        text = _string(row.get("text"), "text", location)
        if query_id in seen:
            raise ValueError(f"{location}: 중복 query_id: {query_id}")
        seen.add(query_id)
        queries.append(Query(query_id, text))
    if not queries:
        raise ValueError(f"{Path(path).name}: 질의가 하나 이상 필요합니다.")
    return tuple(queries)


def load_qrels(
    path: str | Path, queries: tuple[Query, ...], corpus_ids: set[str]
) -> tuple[dict[str, dict[str, float]], int]:
    judgments = {query.query_id: {} for query in queries}
    row_count = 0
    for location, row in _records(Path(path)):
        row_count += 1
        query_id = _string(row.get("query_id"), "query_id", location)
        document_id = _string(row.get("document_id"), "document_id", location)
        relevance = row.get("relevance")
        if (
            isinstance(relevance, bool)
            or not isinstance(relevance, (int, float))
            or not isfinite(relevance)
        ):
            raise ValueError(
                f"{location}: relevance는 bool을 제외한 유한한 숫자여야 합니다."
            )
        if query_id not in judgments:
            raise ValueError(f"{location}: 해당 split에 없는 query_id: {query_id}")
        if document_id not in corpus_ids:
            raise ValueError(f"{location}: corpus에 없는 document_id: {document_id}")
        if (
            document_id in judgments[query_id]
            and judgments[query_id][document_id] != relevance
        ):
            raise ValueError(
                f"{location}: 충돌하는 중복 qrel: {query_id}, {document_id}"
            )
        judgments[query_id][document_id] = relevance
    return judgments, row_count


def validate_dataset(data, raw):
    """Verify the pinned dataset, source hashes and split sizes before an experiment."""
    datasets = {
        split: load_evaluation_dataset(data, split) for split in ("train", "dev")
    }
    manifest = read_json(data / "manifest.json")
    if manifest["dataset_revision"] != PINNED_REVISION:
        raise ValueError("dataset revision mismatch")
    if manifest["settings"] != DATASET_SETTINGS:
        raise ValueError("dataset settings mismatch")
    for name, digest in manifest["source_sha256"].items():
        if file_sha256(raw / PINNED_REVISION / name) != digest:
            raise ValueError("source hash mismatch")
    for split, count in (
        ("train", DATASET_SETTINGS["train_queries"]),
        ("dev", DATASET_SETTINGS["dev_queries"]),
    ):
        if (
            len(datasets[split].documents) != DATASET_SETTINGS["corpus_size"]
            or len(datasets[split].queries) != count
        ):
            raise ValueError("dataset counts mismatch")
    return datasets


def _records(path: Path):
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            location = f"{path.name}:{number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: JSON 형식이 올바르지 않습니다.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{location}: JSON 객체여야 합니다.")
            yield location, row


def _string(value, name: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: {name}는 비어 있지 않은 문자열이어야 합니다.")
    return value


def _check_count(manifest: dict, name: str, actual: int) -> None:
    expected = manifest["output_counts"].get(name)
    if (
        isinstance(expected, bool)
        or not isinstance(expected, int)
        or expected != actual
    ):
        raise ValueError(f"manifest output_counts와 파일 레코드 수가 다릅니다: {name}")

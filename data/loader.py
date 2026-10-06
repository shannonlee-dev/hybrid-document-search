"""원본 및 준비된 JSONL 문서를 공통 문서 형식으로 불러옵니다."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """원본 passage ID와 표시용 텍스트를 보존하는 문서입니다."""

    document_id: str
    text: str
    title: str | None = None


def load_documents(path: str | Path) -> list[Document]:
    """Ko-miracl corpus 형식의 JSONL을 검증하고 문서 목록으로 반환합니다."""
    return _load_documents(path, id_field="_id")


def load_prepared_documents(path: str | Path) -> list[Document]:
    """준비된 document_id 형식의 JSONL을 검증하고 문서 목록으로 반환합니다."""
    return _load_documents(path, id_field="document_id")


def _load_documents(path: str | Path, *, id_field: str) -> list[Document]:
    documents: list[Document] = []
    seen_ids: set[str] = set()

    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            location = f"{Path(path).name}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: JSON 형식이 올바르지 않습니다.") from exc

            if not isinstance(record, dict):
                raise ValueError(f"{location}: JSON 객체여야 합니다.")

            document_id = record.get(id_field)
            text = record.get("text")
            title = record.get("title")

            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError(
                    f"{location}: {id_field}는 비어 있지 않은 문자열이어야 합니다."
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"{location}: text는 비어 있지 않은 문자열이어야 합니다."
                )
            if title is not None and not isinstance(title, str):
                raise ValueError(f"{location}: title은 문자열 또는 null이어야 합니다.")
            if document_id in seen_ids:
                raise ValueError(f"{location}: 중복 document_id: {document_id}")

            if isinstance(title, str) and not title.strip():
                title = None

            documents.append(Document(document_id=document_id, text=text, title=title))
            seen_ids.add(document_id)

    return documents

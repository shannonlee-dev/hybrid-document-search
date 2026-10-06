"""문서 로딩, 검증, 제목 처리와 원문 보존을 검증합니다."""

import json
from pathlib import Path

import pytest

from data.loader import Document, load_documents
from data.preprocess import build_index_text

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ko_miracl_corpus.jsonl"


def write_records(tmp_path, records):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def test_load_fixture():
    documents = load_documents(FIXTURE_PATH)

    assert documents == [
        Document("fixture-001#0", "서울은 대한민국의 수도입니다.", "서울"),
        Document("fixture-002#0", "제주도는 대한민국의 섬입니다.", None),
        Document("fixture-003#0", "한글은 한국어를 표기하는 문자입니다.", None),
        Document("fixture-004#0", "부산은 대한민국의 항구 도시입니다.", None),
    ]


@pytest.mark.parametrize("field", ["_id", "text"])
def test_missing_required_field(tmp_path, field):
    record = {"_id": "doc#0", "text": "본문"}
    del record[field]

    with pytest.raises(ValueError, match=field):
        load_documents(write_records(tmp_path, [record]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("_id", None),
        ("_id", ""),
        ("_id", "   "),
        ("_id", 123),
        ("text", None),
        ("text", ""),
        ("text", "   "),
        ("text", 123),
        ("title", 123),
    ],
)
def test_invalid_field(tmp_path, field, value):
    record = {"_id": "doc#0", "text": "본문"}
    record[field] = value

    with pytest.raises(ValueError, match=field):
        load_documents(write_records(tmp_path, [record]))


def test_duplicate_document_id(tmp_path):
    records = [
        {"_id": "doc#0", "text": "첫 번째 본문"},
        {"_id": "doc#0", "text": "다른 본문"},
    ]

    with pytest.raises(ValueError, match="중복 document_id"):
        load_documents(write_records(tmp_path, records))


def test_invalid_json_reports_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"_id": "doc#0", "text": "본문"}\n{invalid}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="broken.jsonl:2"):
        load_documents(path)


@pytest.mark.parametrize("record", [[], None, "문자열", 123])
def test_non_object_record(tmp_path, record):
    with pytest.raises(ValueError, match="JSON 객체"):
        load_documents(write_records(tmp_path, [record]))


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("서울", "서울\n본문"),
        (None, "본문"),
        ("", "본문"),
        ("   ", "본문"),
    ],
)
def test_build_index_text(title, expected):
    document = Document(document_id="doc#0", text="본문", title=title)

    assert build_index_text(document) == expected


def test_preserve_original_text_and_id(tmp_path):
    text = "  원본 공백\n두 번째 줄  "
    path = write_records(
        tmp_path, [{"_id": "article#12", "text": text, "title": "제목"}]
    )
    document = load_documents(path)[0]

    assert document.document_id == "article#12"
    assert build_index_text(document) == f"제목\n{text}"
    assert document.text == text


def test_skip_blank_lines(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        '\n{"_id": "doc#0", "text": "본문"}\n  \n',
        encoding="utf-8",
    )

    assert load_documents(path) == [Document("doc#0", "본문")]

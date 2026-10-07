"""오프라인 fixture로 subset 준비, 참조 검증 및 재현성을 확인합니다."""

import io
import json

import pytest

from data import preparation
from data.preparation import PINNED_REVISION, file_sha256, prepare_dataset
from scripts.prepare_dataset import main


@pytest.mark.parametrize(
    "options",
    [
        ["--corpus-size", "0"],
        ["--train-queries", "-1"],
        ["--dev-queries", "0"],
        ["--revision", "main"],
    ],
)
def test_cli_rejects_invalid_settings_before_download(monkeypatch, options):
    def unexpected_download(*args):
        pytest.fail("잘못된 설정에서는 다운로드하면 안 됩니다.")

    monkeypatch.setattr("scripts.prepare_dataset.download_sources", unexpected_download)
    with pytest.raises(SystemExit) as exc:
        main(["--download", *options])
    assert exc.value.code == 2


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def raw(tmp_path):
    directory = tmp_path / "raw"
    corpus = [
        {
            "_id": f"passage-{number:02}#0",
            "text": f"  본문 {number}\n둘째 줄  ",
            "title": "제목" if number == 0 else None,
        }
        for number in range(10)
    ]
    corpus[1]["title"] = ""
    corpus[2]["title"] = "   "
    del corpus[4]["title"]
    write_jsonl(directory / "corpus.jsonl", corpus)
    write_jsonl(
        directory / "queries.jsonl",
        [{"_id": str(number), "text": f"질의 {number}"} for number in (1, 2, 3, 4, 99)],
    )
    write_jsonl(
        directory / "qrels/train.jsonl",
        [
            {"query-id": 1, "corpus-id": "passage-00#0", "score": 1},
            {"query-id": 1, "corpus-id": "passage-01#0", "score": 0},
            {"query-id": 3, "corpus-id": "passage-03#0", "score": 1},
            {"query-id": 4, "corpus-id": "passage-04#0", "score": 1},
        ],
    )
    write_jsonl(
        directory / "qrels/dev.jsonl",
        [{"query-id": 2, "corpus-id": "passage-02#0", "score": 1}],
    )
    return directory


def prepare(raw, output, **settings):
    options = {"corpus_size": 5, "train_queries": 2, "dev_queries": 1}
    options.update(settings)
    return prepare_dataset(raw, output, **options)


def test_prepare_common_corpus_and_splits(raw, tmp_path):
    output = tmp_path / "processed"
    manifest = prepare(raw, output)
    corpus = read_jsonl(output / "corpus.jsonl")
    train = read_jsonl(output / "queries_train.jsonl")
    dev = read_jsonl(output / "queries_dev.jsonl")
    train_qrels = read_jsonl(output / "qrels_train.jsonl")

    assert len(corpus) == 5
    assert {row["query_id"] for row in train} == {"1", "4"}
    assert [row["query_id"] for row in dev] == ["2"]
    assert "99" not in {row["query_id"] for row in train + dev}
    by_id = {row["document_id"]: row for row in corpus}
    assert {
        "passage-00#0",
        "passage-01#0",
        "passage-02#0",
        "passage-04#0",
    } <= by_id.keys()
    assert by_id["passage-00#0"]["text"] == "  본문 0\n둘째 줄  "
    assert by_id["passage-00#0"]["title"] == "제목"
    assert all(by_id[f"passage-{number:02}#0"]["title"] is None for number in (1, 2, 4))
    assert any(row["relevance"] == 0 for row in train_qrels)
    assert all(isinstance(row["query_id"], str) for row in train_qrels)
    assert all(set(row) == {"document_id", "text", "title"} for row in corpus)
    assert manifest["source_counts"]["corpus"] == 10
    assert manifest["source_counts"]["queries"] == 5
    assert manifest["dataset_revision"] == PINNED_REVISION

    for name, checksum in manifest["output_sha256"].items():
        assert file_sha256(output / name) == checksum
    for split in ("train", "dev"):
        queries = {
            row["query_id"] for row in read_jsonl(output / f"queries_{split}.jsonl")
        }
        for qrel in read_jsonl(output / f"qrels_{split}.jsonl"):
            assert qrel["query_id"] in queries
            assert qrel["document_id"] in by_id


def test_repeated_runs_are_byte_identical(raw, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    prepare(raw, first)
    prepare(raw, second)

    assert {path.name for path in first.iterdir()} == {
        "corpus.jsonl",
        "queries_train.jsonl",
        "queries_dev.jsonl",
        "qrels_train.jsonl",
        "qrels_dev.jsonl",
        "manifest.json",
    }
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()


def test_corpus_order_does_not_change_selection(raw, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    prepare(raw, first)
    rows = read_jsonl(raw / "corpus.jsonl")
    write_jsonl(raw / "corpus.jsonl", list(reversed(rows)))
    prepare(raw, second)

    assert (first / "corpus.jsonl").read_bytes() == (
        second / "corpus.jsonl"
    ).read_bytes()


def test_required_documents_can_expand_target(raw, tmp_path):
    manifest = prepare(raw, tmp_path / "processed", corpus_size=2)

    assert manifest["selection"]["corpus_size_adjusted"]
    assert manifest["selection"]["effective_corpus_size"] == 4
    assert manifest["output_counts"]["corpus.jsonl"] == 4
    assert manifest["selection"]["background_documents"] == 0


@pytest.mark.parametrize(
    "settings",
    [
        {"corpus_size": 0},
        {"train_queries": -1},
        {"dev_queries": True},
        {"seed": True},
        {"revision": "main"},
    ],
)
def test_invalid_settings(raw, tmp_path, settings):
    with pytest.raises(ValueError):
        prepare(raw, tmp_path / "processed", **settings)


@pytest.mark.parametrize("settings", [{"corpus_size": 11}, {"train_queries": 4}])
def test_insufficient_records(raw, tmp_path, settings):
    output = tmp_path / "processed"
    with pytest.raises(ValueError, match="부족"):
        prepare(raw, output, **settings)
    assert not output.exists()


@pytest.mark.parametrize("which", ["query", "document"])
def test_missing_references(raw, tmp_path, which):
    path = raw / "qrels/train.jsonl"
    rows = read_jsonl(path)
    rows[0]["query-id" if which == "query" else "corpus-id"] = "missing"
    write_jsonl(path, rows)

    with pytest.raises(ValueError, match="query_id|document_id"):
        prepare(raw, tmp_path / "processed")


@pytest.mark.parametrize("which", ["queries.jsonl", "corpus.jsonl"])
def test_duplicate_source_ids(raw, tmp_path, which):
    path = raw / which
    rows = read_jsonl(path)
    write_jsonl(path, rows + [rows[0]])

    with pytest.raises(ValueError, match="중복"):
        prepare(raw, tmp_path / "processed")


@pytest.mark.parametrize("score", [True, "1", float("nan"), float("inf")])
def test_invalid_relevance(raw, tmp_path, score):
    path = raw / "qrels/train.jsonl"
    rows = read_jsonl(path)
    rows[0]["score"] = score
    write_jsonl(path, rows)

    with pytest.raises(ValueError, match="relevance"):
        prepare(raw, tmp_path / "processed")


def test_conflicting_duplicate_qrel(raw, tmp_path):
    path = raw / "qrels/train.jsonl"
    rows = read_jsonl(path)
    write_jsonl(path, rows + [{**rows[0], "score": 0}])

    with pytest.raises(ValueError, match="충돌"):
        prepare(raw, tmp_path / "processed")


def test_identical_duplicate_qrel_is_deduplicated(raw, tmp_path):
    path = raw / "qrels/train.jsonl"
    rows = read_jsonl(path)
    write_jsonl(path, rows + [rows[0]])
    manifest = prepare(raw, tmp_path / "processed")

    assert manifest["source_counts"]["qrels_train"] == 5
    assert manifest["output_counts"]["qrels_train.jsonl"] == 3


def test_overlap_between_splits_is_rejected(raw, tmp_path):
    write_jsonl(
        raw / "qrels/dev.jsonl",
        [{"query-id": 1, "corpus-id": "passage-00#0", "score": 1}],
    )

    with pytest.raises(ValueError, match="train/dev"):
        prepare(raw, tmp_path / "processed")


def test_existing_output_is_not_overwritten(raw, tmp_path):
    output = tmp_path / "processed"
    prepare(raw, output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    with pytest.raises(ValueError, match="비어"):
        prepare(raw, output)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


def test_invalid_json_reports_line(raw, tmp_path):
    path = raw / "corpus.jsonl"
    path.write_text(path.read_text() + "{invalid}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="corpus.jsonl:11"):
        prepare(raw, tmp_path / "processed")


def test_cli_offline(raw, tmp_path, capsys):
    output = tmp_path / "processed"
    result = main(
        [
            "--raw-dir",
            str(raw),
            "--output-dir",
            str(output),
            "--corpus-size",
            "5",
            "--train-queries",
            "2",
            "--dev-queries",
            "1",
        ]
    )

    assert result == 0
    assert "준비 완료" in capsys.readouterr().out
    assert (output / "manifest.json").is_file()


def test_download_is_pinned_and_cached(tmp_path, monkeypatch):
    urls = []

    def fake_urlopen(url, timeout):
        urls.append(url)
        return io.BytesIO(b"fixture\n")

    monkeypatch.setattr(preparation, "urlopen", fake_urlopen)
    directory = preparation.download_sources(tmp_path)
    preparation.download_sources(tmp_path)

    assert directory.name == PINNED_REVISION
    assert len(urls) == 5
    assert all(f"/resolve/{PINNED_REVISION}/" in url for url in urls)
    assert all(
        (directory / name).read_bytes() == b"fixture\n"
        for name in preparation.SOURCE_FILES
    )


def test_failed_download_cleans_temporary_file(tmp_path, monkeypatch):
    def fail(url, timeout):
        raise OSError("network fixture failure")

    monkeypatch.setattr(preparation, "urlopen", fail)
    with pytest.raises(OSError):
        preparation.download_sources(tmp_path)

    assert list((tmp_path / PINNED_REVISION).rglob("*")) == []

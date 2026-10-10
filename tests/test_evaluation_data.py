"""Verify prepared evaluation schemas, splits, corpus references and manifests."""

import json
from pathlib import Path

import pytest

from evaluation.data import (
    file_sha256,
    load_evaluation_dataset,
    load_qrels,
    load_queries,
)

FIXTURES = Path(__file__).parent / "fixtures"


def refresh_manifest(directory):
    counts, hashes = {}, {}
    for path in directory.glob("*.jsonl"):
        counts[path.name] = sum(
            bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
        )
        hashes[path.name] = file_sha256(path)
    manifest = {
        "schema_version": "1",
        "dataset_id": "synthetic-fixture",
        "dataset_revision": "0" * 40,
        "settings": {"seed": 42},
        "output_counts": counts,
        "output_sha256": hashes,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_fixture_dataset_loads_original_fields_and_split(evaluation_directory):
    dev = load_evaluation_dataset(evaluation_directory)
    train = load_evaluation_dataset(evaluation_directory, "train")
    assert len(dev.documents) == 4
    assert dev.documents == train.documents
    assert [query.query_id for query in dev.queries] == ["dev-1", "dev-2"]
    assert [query.query_id for query in train.queries] == ["train-1"]
    assert dev.qrels["dev-1"]["fixture-001#0"] == 1
    assert dev.qrels["dev-1"]["fixture-002#0"] == 0
    assert dev.split == "dev" and train.split == "train"
    assert dev.provenance["query_count"] == 2
    assert dev.provenance["qrel_count"] == 3
    assert dev.provenance["manifest_sha256"] == file_sha256(
        evaluation_directory / "manifest.json"
    )


def test_modified_data_is_rejected_by_manifest(evaluation_directory):
    path = evaluation_directory / "queries_dev.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="체크섬"):
        load_evaluation_dataset(evaluation_directory)


@pytest.mark.parametrize(
    "field", ["dataset_revision", "output_sha256", "output_counts", "schema_version"]
)
def test_invalid_manifest_field(evaluation_directory, field):
    path = evaluation_directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = None
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        load_evaluation_dataset(evaluation_directory)


def test_manifest_counts_and_selected_ids_must_match(evaluation_directory):
    path = evaluation_directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["output_counts"]["queries_dev.jsonl"] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="레코드 수"):
        load_evaluation_dataset(evaluation_directory)
    refresh_manifest(evaluation_directory)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["selected_document_ids"] = ["wrong#0"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="selected_document_ids"):
        load_evaluation_dataset(evaluation_directory)


def test_split_overlap_is_rejected(evaluation_directory):
    path = evaluation_directory / "queries_train.jsonl"
    path.write_text('{"query_id":"dev-1","text":"서울"}\n', encoding="utf-8")
    path = evaluation_directory / "qrels_train.jsonl"
    path.write_text(
        '{"query_id":"dev-1","document_id":"fixture-001#0","relevance":1}\n',
        encoding="utf-8",
    )
    refresh_manifest(evaluation_directory)
    with pytest.raises(ValueError, match="train과 dev"):
        load_evaluation_dataset(evaluation_directory)


@pytest.mark.parametrize(
    "rows, message",
    [
        ("not-json\n", "JSON 형식"),
        ("[]\n", "JSON 객체"),
        ('{"query_id":"","text":"서울"}\n', "query_id"),
        ('{"query_id":"q1","text":" "}\n', "text"),
        ('{"query_id":"q1","text":"서울"}\n{"query_id":"q1","text":"부산"}\n', "중복"),
        ("", "질의"),
    ],
)
def test_invalid_query_records(tmp_path, rows, message):
    path = tmp_path / "queries.jsonl"
    path.write_text(rows, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_queries(path)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"query_id": "train-1"}, "split에 없는"),
        ({"document_id": "missing#0"}, "corpus에 없는"),
        ({"relevance": True}, "relevance"),
        ({"relevance": "1"}, "relevance"),
        ({"relevance": float("nan")}, "relevance"),
        ({"relevance": float("inf")}, "relevance"),
        ({"query_id": 123}, "query_id"),
    ],
)
def test_invalid_qrels(evaluation_directory, changes, message):
    dataset = load_evaluation_dataset(evaluation_directory)
    row = {
        "query_id": "dev-1",
        "document_id": "fixture-001#0",
        "relevance": 1,
        **changes,
    }
    path = evaluation_directory / "invalid-qrels.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_qrels(
            path, dataset.queries, {doc.document_id for doc in dataset.documents}
        )


def test_duplicate_qrels_deduplicate_and_conflicts_fail(evaluation_directory):
    dataset = load_evaluation_dataset(evaluation_directory)
    path = evaluation_directory / "duplicates.jsonl"
    row = {"query_id": "dev-1", "document_id": "fixture-001#0", "relevance": 1}
    path.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
    judgments, count = load_qrels(
        path, dataset.queries, {doc.document_id for doc in dataset.documents}
    )
    assert count == 2
    assert judgments == {"dev-1": {"fixture-001#0": 1}, "dev-2": {}}
    row["relevance"] = 0
    with path.open("a", encoding="utf-8") as source:
        source.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="충돌"):
        load_qrels(
            path, dataset.queries, {doc.document_id for doc in dataset.documents}
        )


def test_empty_qrels_preserve_all_queries(evaluation_directory):
    path = evaluation_directory / "qrels_dev.jsonl"
    path.write_text("", encoding="utf-8")
    refresh_manifest(evaluation_directory)
    assert load_evaluation_dataset(evaluation_directory).qrels == {
        "dev-1": {},
        "dev-2": {},
    }


def test_invalid_split(evaluation_directory):
    with pytest.raises(ValueError, match="split"):
        load_evaluation_dataset(evaluation_directory, "test")


@pytest.mark.parametrize("mutation", [None, "revision", "settings", "source", "counts"])
def test_experiment_dataset_validation(
    evaluation_directory, tmp_path, monkeypatch, mutation
):
    from data.preparation import PINNED_REVISION
    from evaluation import data

    settings = {"corpus_size": 4, "train_queries": 1, "dev_queries": 2, "seed": 42}
    monkeypatch.setattr(data, "DATASET_SETTINGS", settings)
    raw = tmp_path / "raw"
    source = raw / PINNED_REVISION / "corpus.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("source fixture")
    path = evaluation_directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(
        dataset_revision=PINNED_REVISION,
        settings=dict(settings),
        source_sha256={"corpus.jsonl": file_sha256(source)},
    )
    if mutation == "revision":
        manifest["dataset_revision"] = "0" * 40
    elif mutation == "settings":
        manifest["settings"]["seed"] = 1
    elif mutation == "source":
        source.write_text("tampered")
    elif mutation == "counts":
        monkeypatch.setattr(data, "DATASET_SETTINGS", {**settings, "corpus_size": 5})
        manifest["settings"]["corpus_size"] = 5
    path.write_text(json.dumps(manifest))
    if mutation:
        with pytest.raises(ValueError, match=f"{mutation}.*mismatch"):
            data.validate_dataset(evaluation_directory, raw)
    else:
        datasets = data.validate_dataset(evaluation_directory, raw)
        assert set(datasets) == {"train", "dev"}
        assert len(datasets["train"].documents) == 4

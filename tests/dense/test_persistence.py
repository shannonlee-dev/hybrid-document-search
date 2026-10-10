"""Verify Dense index persistence, document mappings and embedding settings."""

import json

import pytest

from retrievers import dense
from retrievers.model_config import MODELS


def test_top_k_and_persistence(dense_documents, model_stub, tmp_path):
    pytest.importorskip("faiss")
    retriever = dense.DenseRetriever.build(
        tuple(dense_documents),
        dense.DenseConfig(model_name="intfloat/multilingual-e5-base", device="cpu"),
    )
    hits = retriever.search("고양이 강아지", top_k=2)
    assert [hit.document_id for hit in hits] == ["dog", "cat"]
    assert [hit.rank for hit in hits] == [1, 2]
    assert [hit.score for hit in hits] == pytest.approx([0.8, 0.6])
    assert hits[0].title == "강아지"
    assert hits[1].title == "고양이"
    assert hits[1].snippet == dense_documents[0].text
    before = retriever.search("고양이 강아지", top_k=10)
    assert len(before) == len(dense_documents)
    assert {hit.document_id for hit in before} == {
        document.document_id for document in dense_documents
    }

    retriever.save(tmp_path)
    restored = dense.DenseRetriever.load(tmp_path, device="cpu")
    assert restored.config == retriever.config
    assert restored.embedder.model is None
    assert restored.search("고양이 강아지", top_k=10) == before
    assert restored.embedder.model.inputs == ["query: 고양이 강아지"]
    assert [restored.get_document(doc.document_id) for doc in dense_documents] == (
        dense_documents
    )


@pytest.mark.parametrize("model_name", MODELS)
def test_saved_revision_survives_default_change(
    model_name, dense_documents, model_stub, tmp_path, monkeypatch
):
    pytest.importorskip("faiss")
    config = dense.DenseConfig(model_name=model_name, revision="a" * 40)
    retriever = dense.DenseRetriever.build(dense_documents, config)
    retriever.save(tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["embedding_config"]["revision"] == "a" * 40
    monkeypatch.setitem(MODELS, model_name, "b" * 40)
    restored = dense.DenseRetriever.load(tmp_path)
    restored.search("고양이", 1)
    assert restored.config.revision == "a" * 40
    assert restored.embedder.model.revision == "a" * 40


@pytest.mark.parametrize("missing", [True, False])
@pytest.mark.parametrize("model_name", MODELS)
def test_unpinned_registered_index_requires_rebuild(
    model_name, missing, dense_documents, model_stub, tmp_path
):
    pytest.importorskip("faiss")
    retriever = dense.DenseRetriever.build(
        dense_documents, dense.DenseConfig(model_name=model_name)
    )
    retriever.save(tmp_path)
    path = tmp_path / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if missing:
        metadata["embedding_config"].pop("revision", None)
    else:
        metadata["embedding_config"]["revision"] = None
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="revision.*rebuild"):
        dense.DenseRetriever.load(tmp_path)


@pytest.mark.parametrize("model_name", ["other/unregistered-model", "/local/model"])
def test_legacy_other_model_index_still_loads(
    model_name, dense_documents, model_stub, tmp_path
):
    pytest.importorskip("faiss")
    config = dense.DenseConfig(model_name=model_name)
    retriever = dense.DenseRetriever.build(dense_documents, config)
    retriever.save(tmp_path)
    path = tmp_path / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["embedding_config"].pop("revision", None)
    path.write_text(json.dumps(metadata), encoding="utf-8")
    restored = dense.DenseRetriever.load(tmp_path)
    restored.search("고양이", 1)
    assert restored.config.revision is None
    assert restored.embedder.model.revision is None


@pytest.mark.parametrize("legacy_device", [None, "cuda:0"])
@pytest.mark.parametrize("device", [None, "cpu"])
def test_load_uses_runtime_device(
    legacy_device, device, dense_documents, model_stub, tmp_path
):
    pytest.importorskip("faiss")
    retriever = dense.DenseRetriever.build(
        dense_documents, dense.DenseConfig(device="cuda:0", batch_size=2)
    )
    before = retriever.search("고양이", 1)
    retriever.save(tmp_path)
    path = tmp_path / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert "device" not in metadata["embedding_config"]
    if legacy_device is not None:
        metadata["embedding_config"]["device"] = legacy_device
        path.write_text(json.dumps(metadata), encoding="utf-8")

    restored = dense.DenseRetriever.load(tmp_path, device=device)
    assert restored.config == dense.DenseConfig(device=device, batch_size=2)
    assert restored.search("고양이", 1) == before
    assert restored.embedder.model.device == device


@pytest.mark.parametrize("corruption", ["mapping", "mapping_type", "config", "version"])
def test_corrupt_metadata_rejected(corruption, dense_documents, model_stub, tmp_path):
    pytest.importorskip("faiss")
    retriever = dense.DenseRetriever.build(dense_documents)
    retriever.save(tmp_path)
    path = tmp_path / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if corruption == "mapping":
        metadata["documents"].pop()
    elif corruption == "mapping_type":
        metadata["documents"] = None
    elif corruption == "config":
        metadata["embedding_config"] = {}
    else:
        metadata["version"] = 2
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="count|mapping|configuration|version"):
        dense.DenseRetriever.load(tmp_path)


def test_raw_scores_and_custom_prefixes_survive_restart(
    dense_documents, model_stub, tmp_path
):
    pytest.importorskip("faiss")
    config = dense.DenseConfig(
        model_name="/local/model",
        normalize_embeddings=False,
        query_prefix="q: ",
        passage_prefix="d: ",
    )
    retriever = dense.DenseRetriever.build(dense_documents, config)
    before = retriever.search("고양이", 1)
    assert before[0].score == 9
    retriever.save(tmp_path)
    restored = dense.DenseRetriever.load(tmp_path)
    assert restored.config == config
    assert restored.search("고양이", 1) == before
    assert restored.embedder.model.inputs == ["q: 고양이"]


def test_checksum_rejects_replaced_index(dense_documents, model_stub, tmp_path):
    pytest.importorskip("faiss")
    retriever = dense.DenseRetriever.build(dense_documents)
    retriever.save(tmp_path)
    (tmp_path / "index.faiss").write_bytes(b"broken index")
    with pytest.raises(ValueError, match="checksum"):
        dense.DenseRetriever.load(tmp_path)


@pytest.mark.parametrize("corruption", ["dimension", "duplicate_id"])
def test_saved_document_contract_rejected(
    corruption, dense_documents, model_stub, tmp_path
):
    pytest.importorskip("faiss")
    retriever = dense.DenseRetriever.build(dense_documents)
    retriever.save(tmp_path)
    path = tmp_path / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if corruption == "dimension":
        metadata["dimension"] += 1
    else:
        metadata["documents"][1]["document_id"] = metadata["documents"][0][
            "document_id"
        ]
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="dimension|unique"):
        dense.DenseRetriever.load(tmp_path)

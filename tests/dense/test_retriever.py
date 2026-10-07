"""Dense Top-K 검색, 공통 검색 계약과 문서 매핑을 확인한다."""

import numpy as np
import pytest

from retrievers import dense


@pytest.mark.parametrize("query", ["", "   ", "\n\t", "\u3000", "\u00a0"])
def test_empty_query_returns_no_hits_without_encoding(
    query, dense_documents, model_stub, tmp_path
):
    pytest.importorskip("faiss")
    retriever = dense.DenseRetriever.build(dense_documents)
    assert retriever.search(query, top_k=3) == []
    retriever.save(tmp_path)
    restored = dense.DenseRetriever.load(tmp_path)
    assert restored.search(query, top_k=3) == []
    assert restored.embedder.model is None


@pytest.mark.parametrize("query", [None, 123])
def test_search_rejects_non_string_query(query, dense_documents, model_stub):
    pytest.importorskip("faiss")
    retriever = dense.DenseRetriever.build(dense_documents)
    with pytest.raises(TypeError, match="query"):
        retriever.search(query, top_k=3)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, "2", None, True])
def test_empty_query_still_requires_positive_top_k(top_k, dense_documents, model_stub):
    pytest.importorskip("faiss")
    retriever = dense.DenseRetriever.build(dense_documents)
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("", top_k=top_k)


def test_build_embeds_shared_indexing_text(dense_documents, model_stub):
    pytest.importorskip("faiss")
    retriever = dense.DenseRetriever.build(
        dense_documents, dense.DenseConfig(device="cpu")
    )
    assert retriever.embedder.model.inputs == [
        "passage: 고양이\n고양이 울음소리는 야옹 입니다.",
        "passage: 강아지\n강아지 울음소리는 멍멍 입니다.",
        "passage: 문서 검색\n문서 검색에는 sparse와 dense 방식이 있습니다.",
    ]
    assert retriever.get_document("cat") == dense_documents[0]


def test_build_rejects_invalid_corpus(dense_documents, model_stub):
    for documents in ([], [dense_documents[0], dense_documents[0]], None):
        with pytest.raises(ValueError):
            dense.DenseRetriever.build(documents)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k(top_k, dense_documents, model_stub):
    pytest.importorskip("faiss")
    retriever = dense.DenseRetriever.build(dense_documents)
    with pytest.raises(ValueError):
        retriever.search("고양이", top_k)


def test_constructor_returns_searchable_snapshot(dense_documents, model_stub):
    pytest.importorskip("faiss")
    embedder = dense.DenseEmbedder(dense.DenseConfig(device="cpu"))
    index = dense.FaissIndex.build(
        embedder.encode_documents(["고양이", "강아지", "검색"])
    )
    retriever = dense.DenseRetriever(
        index=index, documents=dense_documents, embedder=embedder
    )
    dense_documents.clear()
    assert retriever.config is embedder.config
    assert retriever.search("고양이", 1)[0].document_id == "cat"
    assert retriever.get_document("cat").text == "고양이 울음소리는 야옹 입니다."


@pytest.mark.parametrize("corruption", ["empty", "duplicate", "count", "record"])
def test_constructor_rejects_invalid_mapping(corruption, dense_documents):
    pytest.importorskip("faiss")
    index = dense.FaissIndex.build(np.eye(3))
    if corruption == "empty":
        dense_documents.clear()
    elif corruption == "duplicate":
        dense_documents[1] = dense_documents[0]
    elif corruption == "count":
        dense_documents.pop()
    else:
        dense_documents[0] = object()
    with pytest.raises(ValueError, match="documents|unique|count|Document"):
        dense.DenseRetriever(
            index=index, documents=dense_documents, embedder=dense.DenseEmbedder()
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"document_id": "", "text": "text"},
        {"document_id": 12, "text": "text"},
        {"document_id": "a", "text": " "},
        {"document_id": "a", "text": "text", "title": 12},
    ],
)
@pytest.mark.parametrize("construction", ["build", "components"])
def test_retriever_rejects_invalid_document_from_python_api(kwargs, construction):
    from data.loader import Document

    documents = [Document(**kwargs)]
    if construction == "components":
        pytest.importorskip("faiss")
        index = dense.FaissIndex.build([[1, 0, 0]])
    with pytest.raises(ValueError):
        if construction == "build":
            dense.DenseRetriever.build(documents)
        else:
            dense.DenseRetriever(
                index=index, documents=documents, embedder=dense.DenseEmbedder()
            )


def test_document_mapping_is_independent_of_input_list(dense_documents, model_stub):
    pytest.importorskip("faiss")
    retriever = dense.DenseRetriever.build(dense_documents)
    dense_documents.clear()
    assert retriever.search("고양이", 1)[0].document_id == "cat"
    assert retriever.get_document("cat").text == "고양이 울음소리는 야옹 입니다."
    with pytest.raises(KeyError):
        retriever.get_document("missing")

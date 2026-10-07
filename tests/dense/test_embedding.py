"""문서·질의 임베딩, 모델 설정과 입력 검증을 확인한다."""

import numpy as np
import pytest

from retrievers import dense


def test_document_and_query_embeddings(model_stub):
    embedder = dense.DenseEmbedder(dense.DenseConfig(batch_size=2, device="cpu"))
    documents = embedder.encode_documents(["고양이", "강아지"])
    query = embedder.encode_query("고양이")
    assert documents.shape == (2, 3)
    assert query.shape == (1, 3)
    assert documents.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(documents, axis=1), [1, 1])
    np.testing.assert_array_equal(query, documents[:1])
    assert embedder.model.inputs == [
        "passage: 고양이",
        "passage: 강아지",
        "query: 고양이",
    ]
    assert embedder.model.device == "cpu"
    assert embedder.model.options[0]["batch_size"] == 2
    assert embedder.model.options[0]["prompt"] == ""


@pytest.mark.parametrize("name", ["BAAI/bge-m3", "nlpai-lab/KURE-v1"])
def test_other_models_and_raw_inner_product(name, model_stub):
    embedder = dense.DenseEmbedder(
        dense.DenseConfig(model_name=name, normalize_embeddings=False)
    )
    np.testing.assert_array_equal(embedder.encode_query("고양이"), [[3, 0, 0]])
    assert embedder.model.inputs == ["고양이"]


def test_custom_prefixes_for_local_e5(model_stub):
    embedder = dense.DenseEmbedder(
        dense.DenseConfig(
            model_name="/local/model",
            query_prefix="query: ",
            passage_prefix="passage: ",
        )
    )
    embedder.encode_documents(["고양이"])
    embedder.encode_query("고양이")
    assert embedder.model.inputs == ["passage: 고양이", "query: 고양이"]


@pytest.mark.parametrize("text", ["", None])
def test_invalid_query(text, model_stub):
    with pytest.raises(ValueError):
        dense.DenseEmbedder().encode_query(text)


@pytest.mark.parametrize("texts", [[], [""], [None], "고양이"])
def test_invalid_embedding_input(texts, model_stub):
    with pytest.raises(ValueError):
        dense.DenseEmbedder().encode_documents(texts)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": True},
        {"model_name": ""},
        {"normalize_embeddings": "yes"},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        dense.DenseConfig(**kwargs)

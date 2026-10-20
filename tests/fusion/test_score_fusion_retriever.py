"""Real BM25/FAISS integration with synthetic normalized vectors, no model."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("faiss")
pytest.importorskip("bm25s")
pytest.importorskip("sklearn")

from data.loader import Document  # noqa: E402
from fusion.score import ScoreFusionRetriever, score_fusion  # noqa: E402
from indexing.faiss_index import FaissIndex  # noqa: E402
from retrievers.bm25 import BM25Retriever  # noqa: E402
from retrievers.dense import DenseConfig, DenseRetriever  # noqa: E402


class _FixedEmbedder:
    config = DenseConfig(model_name="fixture")

    def __init__(self):
        self.calls = []

    def encode_query(self, query):
        self.calls.append(query)
        return np.array([[1, 0]], dtype=np.float32)


@pytest.fixture
def sources():
    documents = [Document("z", "우주"), Document("a", "고양이")]
    embedder = _FixedEmbedder()
    dense = DenseRetriever(
        index=FaissIndex.build([[0.8, 0.6], [0.8, 0.6]]),
        documents=documents,
        embedder=embedder,
    )
    return BM25Retriever(documents), dense


def test_full_score_array_preserves_corpus_order_and_zero_scores(sources):
    sparse, _ = sources
    assert sparse.documents == (Document("z", "우주"), Document("a", "고양이"))
    scores = sparse.score_all("고양이")
    assert scores[0] == 0
    assert scores[1] == sparse.search("고양이", 1)[0].score
    np.testing.assert_array_equal(sparse.score_all(""), [0, 0])
    np.testing.assert_array_equal(sparse.score_all("zzzzzz"), [0, 0])
    with pytest.raises(TypeError):
        sparse.score_all(None)


def test_fusion_uses_full_scores_and_encodes_query_once(sources):
    sparse, dense = sources
    hybrid = ScoreFusionRetriever(sparse, dense)
    results = hybrid.search("고양이", 1)
    assert results[0].document_id == "a"
    assert results[0].score == pytest.approx(0.85)
    assert results[0].snippet == "고양이"
    assert dense.embedder.calls == ["고양이"]
    reference = score_fusion(dense.search("고양이", 2), sparse.search("고양이", 2), 1)
    assert results == reference


def test_ties_use_document_id_and_dense_index_is_unchanged(sources):
    sparse, dense = sources
    before = dense.index.index.reconstruct_n(0, 2).copy()
    hybrid = ScoreFusionRetriever(sparse, dense, bm25_weight=0)
    results = hybrid.search("고양이", 10)
    assert [r.document_id for r in results] == ["a", "z"]
    assert [r.rank for r in results] == [1, 2]
    np.testing.assert_array_equal(before, dense.index.index.reconstruct_n(0, 2))


def test_invalid_query_top_k_and_empty_query_do_not_encode(sources):
    sparse, dense = sources
    hybrid = ScoreFusionRetriever(sparse, dense)
    assert hybrid.search(" ", 1) == []
    with pytest.raises(ValueError):
        hybrid.search("고양이", 0)
    with pytest.raises(TypeError):
        hybrid.search(None, 1)
    assert dense.embedder.calls == []


def test_no_lexical_matches_keeps_dense_scores(sources):
    sparse, dense = sources
    results = ScoreFusionRetriever(sparse, dense).search("zzzzzz", 10)
    assert [r.document_id for r in results] == ["a", "z"]
    assert [r.score for r in results] == pytest.approx([0.8, 0.8])


@pytest.mark.parametrize(
    ("lexical", "weight", "expected"),
    [
        ([-2, 4], 0.2, [1.0, 0.7]),
        ([-2, 4], 0.0, [0.8, 0.8]),
        ([0, 0], 0.2, [0.8, 0.8]),
        ([-2, -4], 0.2, [0.8, 0.8]),
    ],
)
def test_fusion_formula_has_known_scores(
    sources, monkeypatch, lexical, weight, expected
):
    sparse, dense = sources
    lexical_scores = np.array(lexical, dtype=np.float32)
    monkeypatch.setattr(sparse, "score_all", lambda query: lexical_scores)
    results = ScoreFusionRetriever(sparse, dense, bm25_weight=weight).search("질의", 2)
    assert [r.document_id for r in results] == ["a", "z"]
    assert [r.score for r in results] == pytest.approx(expected)
    np.testing.assert_array_equal(lexical_scores, lexical)


def test_mismatched_document_order_or_text_is_rejected(sources):
    _, dense = sources
    for documents in (
        list(reversed(dense.documents)),
        [Document("z", "다른 본문"), dense.documents[1]],
    ):
        with pytest.raises(ValueError, match="corpus"):
            ScoreFusionRetriever(BM25Retriever(documents), dense)


def test_non_normalized_dense_configuration_is_rejected(sources):
    sparse, dense = sources
    dense.config = DenseConfig(model_name="fixture", normalize_embeddings=False)
    with pytest.raises(ValueError, match="normalized"):
        ScoreFusionRetriever(sparse, dense)


def test_nonfinite_bm25_scores_cannot_silently_disable_fusion(sources, monkeypatch):
    sparse, dense = sources
    hybrid = ScoreFusionRetriever(sparse, dense)
    monkeypatch.setattr(sparse, "score_all", lambda query: np.array([np.nan, 0]))
    with pytest.raises(ValueError, match="finite"):
        hybrid.search("고양이", 1)


def test_winner_outside_both_top_one_lists_is_not_lost():
    documents = [
        Document("a", "우주"),
        Document("b", "고양이 동물"),
        Document("c", "고양이"),
    ]
    cosines = np.array([0.8, 0.79, 0.1], dtype=np.float32)
    dense = DenseRetriever(
        index=FaissIndex.build(np.column_stack([cosines, np.sqrt(1 - cosines**2)])),
        documents=documents,
        embedder=_FixedEmbedder(),
    )
    sparse = BM25Retriever(documents)
    assert dense.search("고양이", 1)[0].document_id == "a"
    assert sparse.search("고양이", 1)[0].document_id == "c"
    assert ScoreFusionRetriever(sparse, dense).search("고양이", 1)[0].document_id == "b"


@pytest.mark.parametrize("vector", [[[2, 0]], [[0.5, 0]]])
def test_query_must_be_normalized_even_if_config_says_it_is(
    sources, monkeypatch, vector
):
    sparse, dense = sources
    monkeypatch.setattr(dense.embedder, "encode_query", lambda query: vector)
    with pytest.raises(ValueError, match="normalized query"):
        ScoreFusionRetriever(sparse, dense).search("고양이", 1)


def test_wrong_bm25_array_length_cannot_broadcast(sources, monkeypatch):
    sparse, dense = sources
    monkeypatch.setattr(sparse, "score_all", lambda query: np.array([1.0]))
    with pytest.raises(ValueError, match="corpus"):
        ScoreFusionRetriever(sparse, dense).search("고양이", 1)


@pytest.mark.parametrize("weight", [0, 0.025, 0.05, 0.1, 2])
def test_random_full_corpus_paths_agree_with_faiss_and_scalar_formula(weight):
    from retrievers.base import SearchResult

    rng = np.random.default_rng(17)
    documents = [Document(f"d{i:02}", "고양이" if i % 2 else "우주") for i in range(31)]
    rng.shuffle(documents)
    vectors = rng.normal(size=(len(documents), 17)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query_vector = rng.normal(size=(1, 17)).astype(np.float32)
    query_vector /= np.linalg.norm(query_vector)
    embedder = _FixedEmbedder()
    embedder.encode_query = lambda query: query_vector
    dense = DenseRetriever(
        index=FaissIndex.build(vectors), documents=documents, embedder=embedder
    )
    sparse = BM25Retriever(documents)
    # 순수 함수 API는 음수 점수와 모든 점수가 0 이하인 입력도 허용한다.
    for lexical in (
        rng.uniform(-2, 3, len(documents)),
        np.zeros(len(documents)),
        -rng.uniform(0, 2, len(documents)),
    ):
        sparse.score_all = lambda query, scores=lexical: scores
        full_dense = dense.search("질의", len(documents))
        by_id = {r.document_id: r.score for r in full_dense}
        for doc, score in zip(documents, vectors @ query_vector[0], strict=True):
            assert by_id[doc.document_id] == pytest.approx(float(score), abs=1e-7)
        full_sparse = [
            SearchResult(d.document_id, i + 1, float(s))
            for i, (d, s) in enumerate(zip(documents, lexical, strict=True))
        ]
        for top_k in (1, 10, 100):
            actual = ScoreFusionRetriever(sparse, dense, bm25_weight=weight).search(
                "질의", top_k
            )
            expected = score_fusion(
                full_dense[::-1], full_sparse[::-1], top_k, bm25_weight=weight
            )
            assert [r.document_id for r in actual] == [r.document_id for r in expected]
            assert [r.score for r in actual] == pytest.approx(
                [r.score for r in expected], abs=1e-7
            )
            if weight == 0:
                assert [r.document_id for r in actual] == [
                    r.document_id for r in dense.search("질의", top_k)
                ]


def test_vector_snapshot_is_independent_of_later_faiss_mutation(sources):
    sparse, dense = sources
    hybrid = ScoreFusionRetriever(sparse, dense)
    expected = hybrid.search("고양이", 2)
    dense_before = dense.search("고양이", 2)
    snapshot = dense.index.snapshot_vectors()
    snapshot[:] = 0
    assert dense.search("고양이", 2) == dense_before
    dense.index.index.reset()
    dense.index.index.add(np.array([[-1, 0], [0, -1]], dtype=np.float32))
    assert hybrid.search("고양이", 2) == expected


@pytest.mark.parametrize("query", ["고양이", "우주", "", " ", "unknown"])
def test_bm25_full_array_matches_existing_positive_hit_policy(sources, query):
    sparse, _ = sources
    scores = sparse.score_all(query)
    assert scores.shape == (len(sparse.documents),)
    assert scores.dtype == np.float32
    expected = [
        (d.document_id, float(s))
        for d, s in zip(sparse.documents, scores, strict=True)
        if s > 0
    ]
    expected.sort(key=lambda hit: -hit[1])
    assert [(r.document_id, r.score) for r in sparse.search(query, 100)] == expected


def test_weight_variants_share_frozen_snapshot_without_changing_original(sources):
    sparse, dense = sources
    original = ScoreFusionRetriever(sparse, dense, bm25_weight=0)
    variant = original.with_bm25_weight(0.1)
    assert np.shares_memory(original._vectors, variant._vectors)
    with pytest.raises(ValueError):
        variant._vectors[0, 0] = 0
    assert original.search("고양이", 1)[0].score == pytest.approx(0.8)
    assert variant.search("고양이", 1)[0].score == pytest.approx(0.9)
    dense.index.index.reset()
    assert variant.search("고양이", 1)[0].score == pytest.approx(0.9)
    with pytest.raises(ValueError):
        original.with_bm25_weight(float("nan"))

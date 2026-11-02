"""Fixed alpha union min-max retriever must match the Finance evaluation fusion."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("faiss")
pytest.importorskip("bm25s")
pytest.importorskip("sklearn")

from data.loader import Document  # noqa: E402
from evaluation.query_encoder import fixed_fusion_rankings  # noqa: E402
from fusion.minmax import UnionMinMaxFusionRetriever  # noqa: E402
from indexing.faiss_index import FaissIndex  # noqa: E402
from retrievers.bm25 import BM25Retriever  # noqa: E402
from retrievers.dense import DenseConfig, DenseRetriever  # noqa: E402

WORDS = [
    "카드",
    "해지",
    "연회비",
    "포인트",
    "결제",
    "한도",
    "대출",
    "이자",
    "환불",
    "분실",
]


class _Embedder:
    config = DenseConfig(model_name="fixture")

    def __init__(self, vector):
        self.vector = vector

    def encode_query(self, query):
        return self.vector[None, :]


def _sources(seed=0, count=300, dim=8):
    rng = np.random.default_rng(seed)
    documents = [
        Document(f"d{i:03d}", " ".join(rng.choice(WORDS, size=4))) for i in range(count)
    ]
    vectors = rng.normal(size=(count, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = rng.normal(size=dim).astype(np.float32)
    query /= np.linalg.norm(query)
    dense = DenseRetriever(
        index=FaissIndex.build(vectors), documents=documents, embedder=_Embedder(query)
    )
    return BM25Retriever(documents), dense, vectors, query


@pytest.mark.parametrize("alpha", [0.0, 0.5, 0.84, 0.89, 1.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_fixed_fusion_rankings(alpha, seed):
    sparse, dense, vectors, query = _sources(seed)
    fusion = UnionMinMaxFusionRetriever(sparse, dense, alpha=alpha, pool_k=20)
    text = "카드 해지 연회비"
    dense_scores = vectors.astype(np.float64) @ query.astype(np.float64)
    sparse_scores = np.asarray(sparse.score_all(text), dtype=np.float64)
    expected = fixed_fusion_rankings(dense_scores, sparse_scores, [alpha], pool_k=20)[0]
    rows, _ = fusion.rank(fusion.dense_scores(query), sparse_scores, 10)
    np.testing.assert_array_equal(rows, expected)
    hits = fusion.search(text, 10)
    assert [h.document_id for h in hits] == [f"d{r:03d}" for r in expected]
    assert [h.rank for h in hits] == list(range(1, 11))


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), True])
def test_invalid_alpha_is_rejected(alpha):
    sparse, dense, *_ = _sources()
    with pytest.raises(ValueError):
        UnionMinMaxFusionRetriever(sparse, dense, alpha=alpha)


def test_corpus_mismatch_is_rejected():
    sparse, dense, *_ = _sources()
    other = BM25Retriever([Document("x", "카드")])
    with pytest.raises(ValueError, match="same ordered corpus"):
        UnionMinMaxFusionRetriever(other, dense, alpha=0.5)


def test_blank_query_returns_nothing():
    sparse, dense, *_ = _sources()
    assert UnionMinMaxFusionRetriever(sparse, dense, alpha=0.84).search("  ", 5) == []

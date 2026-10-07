"""실제 FAISS로 벡터 검증, 내적 검색, 저장 형식 호환성을 검증한다."""

import numpy as np
import pytest

from indexing.faiss_index import FaissIndex

faiss = pytest.importorskip("faiss")


@pytest.mark.parametrize(
    "vectors", [[], [1, 2], [[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]]]
)
def test_invalid_build_vectors(vectors):
    with pytest.raises(ValueError):
        FaissIndex.build(vectors)


def test_top_k_rows_and_vector_dimensions():
    index = FaissIndex.build(np.eye(2))
    assert index.count == 2
    assert index.dimension == 2
    assert index.search([[0, 1]], 1) == [(1, 1.0)]
    assert index.search([[0, 1]], 10) == [(1, 1.0), (0, 0.0)]
    with pytest.raises(ValueError):
        index.search([[1, 0, 0]], 1)
    with pytest.raises(ValueError):
        index.search(np.eye(2), 1)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k(top_k):
    index = FaissIndex.build([[1, 0]])
    with pytest.raises(ValueError):
        index.search([[1, 0]], top_k)


def test_vector_index_round_trip(tmp_path):
    index = FaissIndex.build([[1, 0], [0, 2]])
    path = tmp_path / "nested" / "index.faiss"
    index.save(path)
    restored = FaissIndex.load(path)
    assert restored.dimension == 2
    assert restored.count == 2
    assert restored.search([[0, 1]], 2) == [(1, 2.0), (0, 0.0)]


def test_rejects_incompatible_saved_index(tmp_path):
    path = tmp_path / "index.faiss"
    index = faiss.IndexFlatL2(2)
    index.add(np.array([[1, 0]], dtype=np.float32))
    faiss.write_index(index, str(path))
    with pytest.raises(ValueError, match="type"):
        FaissIndex.load(path)


def test_rejects_empty_saved_index(tmp_path):
    path = tmp_path / "index.faiss"
    faiss.write_index(faiss.IndexFlatIP(2), str(path))
    with pytest.raises(ValueError, match="empty"):
        FaissIndex.load(path)

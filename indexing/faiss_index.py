"""벡터의 내적 검색과 FAISS 파일 저장을 담당한다."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class FaissIndex:
    """IndexFlatIP를 사용하며 정규화와 행별 문서 매핑은 호출자가 책임진다."""

    def __init__(self, index):
        import faiss

        if not isinstance(index, faiss.IndexFlatIP):
            raise ValueError("unsupported saved index type")
        if index.ntotal == 0:
            raise ValueError("index must not be empty")
        self.index = index

    @property
    def dimension(self) -> int:
        return self.index.d

    @property
    def count(self) -> int:
        return self.index.ntotal

    @classmethod
    def build(cls, vectors) -> "FaissIndex":
        import faiss

        vectors = _as_vectors(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index)

    @classmethod
    def load(cls, path: str | Path) -> "FaissIndex":
        """벡터 인덱스만 복원한다. 문서 매핑과 임베딩 설정은 호출자가 검증한다."""
        import faiss

        try:
            index = faiss.read_index(str(path))
        except RuntimeError as exc:
            raise ValueError("invalid saved FAISS index") from exc
        return cls(index)

    def search(self, query_vector, top_k: int) -> list[tuple[int, float]]:
        _validate_top_k(top_k)
        query_vector = _as_vectors(query_vector, self.dimension)
        if len(query_vector) != 1:
            raise ValueError("search requires exactly one query vector")
        scores, rows = self.index.search(query_vector, min(top_k, self.count))
        return [
            (int(row), float(score))
            for row, score in zip(rows[0], scores[0], strict=True)
        ]

    def save(self, path: str | Path) -> None:
        import faiss

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))


def _as_vectors(vectors, dimension: int | None = None) -> "np.ndarray":
    """유한한 비영벡터를 검증하고 FAISS 입력에 필요한 연속 float32 배열로 변환한다."""
    import numpy as np

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or not all(array.shape):
        raise ValueError("vectors must be a non-empty 2D matrix")
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError("embedding dimension does not match the index")
    if not np.isfinite(array).all() or (np.linalg.norm(array, axis=1) == 0).any():
        raise ValueError("vectors must be finite and nonzero")
    return np.ascontiguousarray(array)


def _validate_top_k(top_k: int) -> None:
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

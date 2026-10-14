"""Search vectors by inner product and persist FAISS indexes."""

from pathlib import Path
from typing import TYPE_CHECKING

from indexing.vectors import as_vectors

if TYPE_CHECKING:
    import numpy as np


class FaissIndex:
    """Wrap IndexFlatIP; callers manage normalization and row-to-document mappings."""

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

    def snapshot_vectors(self) -> "np.ndarray":
        """Copy IndexFlatIP vectors in row order, independent of later mutations."""
        return as_vectors(self.index.reconstruct_n(0, self.count), self.dimension)

    @classmethod
    def build(cls, vectors) -> "FaissIndex":
        import faiss

        vectors = as_vectors(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index)

    @classmethod
    def load(cls, path: str | Path) -> "FaissIndex":
        """Restore vectors; callers validate document mappings and embedding settings."""
        import faiss

        try:
            index = faiss.read_index(str(path))
        except RuntimeError as exc:
            raise ValueError("invalid saved FAISS index") from exc
        return cls(index)

    def search(self, query_vector, top_k: int) -> list[tuple[int, float]]:
        _validate_top_k(top_k)
        query_vector = as_vectors(query_vector, self.dimension)
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


def _validate_top_k(top_k: int) -> None:
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

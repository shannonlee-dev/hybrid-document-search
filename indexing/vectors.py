"""Shared vector validation for embedding, indexing, and score fusion."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def as_vectors(vectors, dimension: int | None = None) -> "np.ndarray":
    """Validate finite, nonzero vectors and return a contiguous float32 matrix."""
    import numpy as np

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or not all(array.shape):
        raise ValueError("vectors must be a non-empty 2D matrix")
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError("embedding dimension does not match the index")
    if not np.isfinite(array).all() or (np.linalg.norm(array, axis=1) == 0).any():
        raise ValueError("vectors must be finite and nonzero")
    return np.ascontiguousarray(array)

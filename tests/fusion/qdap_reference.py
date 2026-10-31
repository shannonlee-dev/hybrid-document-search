"""Independent scalar reference for vectorized QDAP target tests."""

import numpy as np

from evaluation.qdap import PoolBatch, rank_pools, ranking_metrics


def graded_utilities(pools: PoolBatch, judgments, *, top_k=10):
    """Build all 101 supervised nDCG utilities using full-qrels IDCG."""
    curves = []
    for alpha in np.linspace(0.0, 1.0, 101):
        rankings = rank_pools(pools, np.full(len(pools.indices), alpha), top_k=top_k)
        _, values = ranking_metrics(rankings, judgments)
        curves.append(values[:, 0])
    return np.column_stack(curves)

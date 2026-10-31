"""Check full-qrels targets and legacy-vs-paper fusion contracts."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")


def test_fast_targets_equal_reference_with_missing_relevant_and_padding():
    from evaluation.qdap import PoolBatch
    from evaluation.qdap_benchmark import batch_utilities
    from tests.fusion.qdap_reference import graded_utilities

    pools = PoolBatch.from_scores(
        [[0.9, 0.2, 0.8], [0.1, 0.7, 0.2]],
        [[0.1, 0.9, 0.8], [0.2, 0.8, 0.3]],
        ["z", "a", "c"],
        depth=1,
    )
    qrels = [{0: 3, 1: 1}, {0: 3, 1: 2}]
    np.testing.assert_allclose(
        batch_utilities(pools, qrels), graded_utilities(pools, qrels)
    )


def test_legacy_fixed_uses_full_corpus_max_and_document_id_ties():
    from evaluation.qdap_benchmark import legacy_ranking

    ids = np.array(["z", "a", "c"])
    np.testing.assert_array_equal(
        legacy_ranking([1, 1, 0], [0, 0, 1], ids, 0.05), [1, 0, 2]
    )

"""Score fusion must preserve real scores and consider the complete corpus."""

import pytest

from fusion.score import score_fusion
from retrievers.base import SearchResult


def _hit(document_id, score, rank=1):
    return SearchResult(document_id, rank, score, title=document_id)


def test_raw_scores_and_global_bm25_max_determine_ranking():
    dense = [_hit("a", 0.80), _hit("b", 0.79), _hit("c", -0.2)]
    sparse = [_hit("c", 100), _hit("b", 50)]
    results = score_fusion(dense, sparse, top_k=2, bm25_weight=0.05)
    assert [r.document_id for r in results] == ["b", "a"]
    assert [r.score for r in results] == pytest.approx([0.815, 0.8])
    assert [r.rank for r in results] == [1, 2]
    assert results[0].title == "b"


@pytest.mark.parametrize("sparse", [[], [_hit("a", 0)], [_hit("a", -2)]])
def test_nonpositive_bm25_max_keeps_dense_scores(sparse):
    results = score_fusion([_hit("a", -0.5)], sparse, top_k=10)
    assert results == [_hit("a", -0.5)]


def test_zero_weight_and_ties_are_independent_of_input_order():
    dense = [_hit("z", 0.5), _hit("a", 0.5)]
    for items in [dense, dense[::-1]]:
        results = score_fusion(items, [_hit("z", 10)], top_k=20, bm25_weight=0)
        assert [r.document_id for r in results] == ["a", "z"]
        assert [r.rank for r in results] == [1, 2]


def test_signed_bm25_scores_use_global_max_and_missing_hits_contribute_zero():
    results = score_fusion(
        [_hit("a", 0.8), _hit("b", 0.8), _hit("c", 0.8)],
        [_hit("a", -2), _hit("b", 4)],
        top_k=3,
        bm25_weight=0.2,
    )
    assert [r.document_id for r in results] == ["b", "c", "a"]
    assert [r.score for r in results] == pytest.approx([1.0, 0.8, 0.7])


@pytest.mark.parametrize("side", ["dense", "sparse"])
def test_duplicate_document_in_one_source_is_rejected(side):
    dense, sparse = [_hit("a", 0.5)], [_hit("a", 1)]
    (dense if side == "dense" else sparse).append(_hit("a", 0.2))
    with pytest.raises(ValueError, match="duplicate"):
        score_fusion(dense, sparse, top_k=1)


def test_sparse_document_without_dense_score_is_rejected():
    with pytest.raises(ValueError, match="Dense"):
        score_fusion([_hit("a", 0.5)], [_hit("b", 1)], top_k=1)


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), "0.05"])
def test_invalid_weights_are_rejected(value):
    with pytest.raises(ValueError):
        score_fusion([], [], top_k=1, bm25_weight=value)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_top_k_is_rejected(value):
    with pytest.raises(ValueError):
        score_fusion([], [], top_k=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1"])
@pytest.mark.parametrize("side", ["dense", "sparse"])
def test_invalid_scores_are_rejected(value, side):
    dense = [_hit("a", value if side == "dense" else 0.5)]
    sparse = [_hit("a", value if side == "sparse" else 1)]
    with pytest.raises(ValueError):
        score_fusion(dense, sparse, top_k=1)

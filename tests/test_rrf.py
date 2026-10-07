"""Offline tests for rank-based RRF using synthetic SearchResult lists."""

import pytest

from fusion.rrf import DEFAULT_RANK_CONSTANT, reciprocal_rank_fusion
from retrievers.base import SearchResult


def hit(doc_id, rank, score=0.0, title=None, snippet=None):
    return SearchResult(
        document_id=doc_id, rank=rank, score=score, title=title, snippet=snippet
    )


def rrf(k, rank):
    return 1.0 / (k + rank)


def test_default_rank_constant_is_60():
    assert DEFAULT_RANK_CONSTANT == 60


def test_single_source_keeps_order_and_scores_by_rank():
    fused = reciprocal_rank_fusion([[hit("A", 1), hit("B", 2)]], top_k=5)

    assert [r.document_id for r in fused] == ["A", "B"]
    assert fused[0].score == pytest.approx(rrf(60, 1))
    assert fused[1].score == pytest.approx(rrf(60, 2))


def test_disjoint_sources_are_interleaved_by_rrf_score():
    sparse = [hit("A", 1), hit("B", 2)]
    dense = [hit("C", 1), hit("D", 2)]

    fused = reciprocal_rank_fusion([sparse, dense], top_k=10)

    # A and C tie on score; document_id breaks the tie.
    assert [r.document_id for r in fused] == ["A", "C", "B", "D"]


def test_overlapping_document_is_reported_once_with_summed_score():
    sparse = [hit("A", 1), hit("B", 2)]
    dense = [hit("B", 1), hit("C", 2)]

    fused = reciprocal_rank_fusion([sparse, dense], top_k=10)

    assert [r.document_id for r in fused].count("B") == 1
    b = next(r for r in fused if r.document_id == "B")
    assert b.score == pytest.approx(rrf(60, 2) + rrf(60, 1))
    assert fused[0].document_id == "B"


def test_final_ranks_are_contiguous_from_one():
    fused = reciprocal_rank_fusion(
        [[hit("A", 1), hit("B", 2), hit("C", 3)], [hit("D", 1)]], top_k=10
    )
    assert [r.rank for r in fused] == [1, 2, 3, 4]


def test_top_k_truncates_after_fusion():
    fused = reciprocal_rank_fusion(
        [[hit("A", 1), hit("B", 2), hit("C", 3)], [hit("D", 1)]], top_k=2
    )
    assert [r.document_id for r in fused] == ["A", "D"]
    assert [r.rank for r in fused] == [1, 2]


def test_empty_source_lists_are_valid():
    fused = reciprocal_rank_fusion([[], [hit("A", 1)]], top_k=5)
    assert [r.document_id for r in fused] == ["A"]


def test_all_empty_sources_return_empty():
    assert reciprocal_rank_fusion([[], []], top_k=5) == []
    assert reciprocal_rank_fusion([], top_k=5) == []


def test_raw_scores_are_ignored_when_ranks_are_preserved():
    ranks = [[("A", 1), ("B", 2), ("C", 3)], [("C", 1), ("D", 2)]]
    low = [[hit(d, r, score=0.1) for d, r in lst] for lst in ranks]
    high = [[hit(d, r, score=999.0 - r) for d, r in lst] for lst in ranks]

    assert reciprocal_rank_fusion(low, top_k=10) == reciprocal_rank_fusion(
        high, top_k=10
    )


def test_equal_scores_are_ordered_deterministically():
    # X and Y each get 1/61 from one source at rank 1; order is by document_id.
    fused_a = reciprocal_rank_fusion([[hit("Y", 1)], [hit("X", 1)]], top_k=2)
    fused_b = reciprocal_rank_fusion([[hit("X", 1)], [hit("Y", 1)]], top_k=2)

    assert [r.document_id for r in fused_a] == ["X", "Y"]
    assert [r.document_id for r in fused_b] == ["X", "Y"]


def test_equal_score_prefers_better_best_rank_before_document_id():
    # With rank_constant=0: X at rank 1 scores 1/1 = 1.0, and Y at rank 2 in two
    # sources scores 1/2 + 1/2 = 1.0 (exact in float). Y has the smaller ID, so
    # only the best-rank rule can put X first.
    sources = [[hit("X", 1), hit("Y", 2)], [hit("Y", 2)]]
    fused = reciprocal_rank_fusion(sources, top_k=2, rank_constant=0)

    assert [r.document_id for r in fused] == ["X", "Y"]
    assert fused[0].score == fused[1].score == pytest.approx(1.0)


def test_metadata_comes_from_best_ranked_occurrence():
    sparse = [hit("A", 1), hit("B", 2, title="sparse-title", snippet="sparse-snip")]
    dense = [hit("X", 1), hit("B", 1, title="dense-title", snippet="dense-snip")]

    fused = reciprocal_rank_fusion([sparse, dense], top_k=10)
    b = next(r for r in fused if r.document_id == "B")

    assert (b.title, b.snippet) == ("dense-title", "dense-snip")


def test_metadata_tie_prefers_earlier_source():
    first = [hit("A", 1, title="first", snippet="first-snip")]
    second = [hit("A", 1, title="second", snippet="second-snip")]

    fused = reciprocal_rank_fusion([first, second], top_k=5)

    assert (fused[0].title, fused[0].snippet) == ("first", "first-snip")


def test_metadata_is_not_mixed_across_occurrences():
    sparse = [hit("A", 1, title="T", snippet=None)]
    dense = [hit("A", 3, title=None, snippet="S")]

    fused = reciprocal_rank_fusion([sparse, dense], top_k=5)

    assert (fused[0].title, fused[0].snippet) == ("T", None)


def test_missing_metadata_stays_none():
    fused = reciprocal_rank_fusion([[hit("A", 1)], [hit("A", 2)]], top_k=5)
    assert (fused[0].title, fused[0].snippet) == (None, None)


@pytest.mark.parametrize("bad_rank", [0, -1])
def test_non_positive_rank_is_rejected(bad_rank):
    with pytest.raises(ValueError, match="rank"):
        reciprocal_rank_fusion([[hit("A", bad_rank)]], top_k=5)


def test_bool_rank_is_rejected():
    with pytest.raises(ValueError, match="rank"):
        reciprocal_rank_fusion([[hit("A", True)]], top_k=5)


def test_duplicate_document_id_within_one_source_is_rejected():
    with pytest.raises(ValueError, match="duplicate document_id"):
        reciprocal_rank_fusion([[hit("A", 1), hit("A", 2)]], top_k=5)


@pytest.mark.parametrize("bad_top_k", [0, -1, True, 1.0, "3"])
def test_invalid_top_k_is_rejected(bad_top_k):
    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fusion([[hit("A", 1)]], top_k=bad_top_k)


@pytest.mark.parametrize("bad_k", [-1, True, 1.5])
def test_invalid_rank_constant_is_rejected(bad_k):
    with pytest.raises(ValueError, match="rank_constant"):
        reciprocal_rank_fusion([[hit("A", 1)]], top_k=5, rank_constant=bad_k)


def test_custom_rank_constant_changes_scores():
    fused = reciprocal_rank_fusion([[hit("A", 1)]], top_k=5, rank_constant=0)
    assert fused[0].score == pytest.approx(1.0)

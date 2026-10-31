"""Exercise contextual routing without model downloads or accelerator access."""

import pytest

from data.loader import Document

np = pytest.importorskip("numpy")


def test_odds_weight_keeps_dense_sparse_orientation_and_handles_extremes():
    from fusion.contextual import relevance_weight

    assert relevance_weight([0.8, 0.2]) == pytest.approx(16 / 17)
    assert relevance_weight([0.2, 0.8]) == pytest.approx(1 / 17)
    assert relevance_weight([0, 0]) == 0.5
    assert 0 <= relevance_weight([0, 1]) < 1e-10
    with pytest.raises(ValueError):
        relevance_weight([np.nan, 0.5])


def test_union_minmax_and_ties_preserve_corpus_order():
    from fusion.contextual import fuse_scores

    # 두 채널의 top1은 다르지만 가중합은 동점이므로 corpus 순서로 결정한다.
    indices, scores = fuse_scores([5, 3, 1], [0, 1, 2], alpha=0.5, top_k=3)
    assert indices.tolist() == [0, 1, 2]
    np.testing.assert_allclose(scores, [0.5, 0.5, 0.5])


def test_agreement_skips_judge_and_uses_fixed_weight():
    from fusion.contextual import ContextualRetriever, FixedPolicy

    calls = []
    documents = [Document("a", "first"), Document("b", "second")]
    retriever = ContextualRetriever(
        documents,
        np.eye(2),
        encode=lambda query: np.array([0.9, 0.1]),
        sparse_scores=lambda query: np.array([1.0, 0.0]),
        judge=lambda query, texts: calls.append(texts),
        fixed=FixedPolicy(alpha=0.84),
    )
    run = retriever.search_detailed("query", 2)
    assert calls == []
    assert run.alpha == 0.84
    assert run.judged is False
    assert [hit.document_id for hit in run.results] == ["a", "b"]
    assert [hit.rank for hit in run.results] == [1, 2]
    assert run.elapsed_ms["total"] == pytest.approx(
        sum(run.elapsed_ms[k] for k in ("encode", "retrieve", "judge", "fusion"))
    )


def test_disagreement_reads_actual_dense_then_sparse_candidate_texts():
    from fusion.contextual import ContextualRetriever

    calls = []

    def judge(query, texts):
        calls.append((query, texts))
        return [0.1, 0.9]

    retriever = ContextualRetriever(
        [Document("a", "dense body"), Document("b", "sparse body")],
        np.eye(2),
        encode=lambda query: np.array([1.0, 0.0]),
        sparse_scores=lambda query: np.array([0.0, 1.0]),
        judge=judge,
    )
    run = retriever.search_detailed("card fee", 2)
    assert calls == [("card fee", ["dense body", "sparse body"])]
    assert run.results[0].document_id == "b"
    assert run.judged is True
    assert run.alpha == pytest.approx(1 / 82)


@pytest.mark.parametrize("alpha", [-1, 2, float("nan")])
def test_invalid_policy_rejected(alpha):
    from fusion.contextual import FixedPolicy

    with pytest.raises(ValueError):
        FixedPolicy(alpha=alpha)


def test_top100_union_handles_small_and_constant_channels():
    from fusion.contextual import fuse_scores

    indices, scores = fuse_scores([1, 1], [0, 0], alpha=0.3, top_k=10)
    assert indices.tolist() == [0, 1]
    assert scores.tolist() == [0, 0]
    with pytest.raises(ValueError):
        fuse_scores([1, 2], [1], alpha=0.5)

"""Exercise QDAP heads, loss, readouts, and candidate ranking."""

import math

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from evaluation.qdap import (  # noqa: E402
    PoolBatch,
    probability_alphas,
    rank_pools,
)
from fusion.qdap import (  # noqa: E402
    QDAPSmall,
    distribution_targets,
    qdap_loss,
)
from tests.fusion.qdap_reference import graded_utilities  # noqa: E402


def test_architecture_detaches_frozen_embeddings_and_updates_head():
    torch.manual_seed(7)
    model = QDAPSmall(4)
    x = torch.randn(3, 4, requires_grad=True)
    before = model.linear.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    prediction = model(x)
    assert prediction.shape == (3, 101)
    assert model.smooth.kernel_size == (7,)
    torch.testing.assert_close(prediction.sum(-1), torch.ones(3))
    qdap_loss(prediction, distribution_targets(torch.randn(3, 101))).backward()
    optimizer.step()
    assert x.grad is None
    assert not torch.equal(before, model.linear.weight)


def test_loss_uses_explicit_ce_interpretation_and_unnormalized_cdf_distance():
    target = torch.zeros(1, 101)
    target[0, :2] = torch.tensor([0.25, 0.75])
    prediction = torch.full((1, 101), 1 / 101)
    wd = sum(abs((j + 1) / 101 - (0.25 if j == 0 else 1)) for j in range(101))
    literal = 0.62 * (0.25**2 + 0.75**2) * math.log(101) + 0.38 * wd
    reference = 0.62 * math.log(101) + 0.38 * wd
    assert qdap_loss(prediction, target, ce_mode="literal").item() == pytest.approx(
        literal, rel=1e-5
    )
    assert qdap_loss(prediction, target, ce_mode="reference").item() == pytest.approx(
        reference, rel=1e-5
    )
    with pytest.raises(ValueError, match="ce_mode"):
        qdap_loss(prediction, target, ce_mode="other")


def test_readouts_keep_expectation_and_first_maximum_distinct():
    p = np.zeros((2, 101))
    p[0, 10] = 0.5
    p[0, 90] = 0.5
    p[1, 65] = 1.0
    np.testing.assert_allclose(probability_alphas(p, "mean"), [0.5, 0.65])
    np.testing.assert_allclose(probability_alphas(p, "argmax"), [0.1, 0.65])
    with pytest.raises(ValueError, match="readout"):
        probability_alphas(p, "median")


def test_pool_union_minmax_id_ties_and_padding_are_label_free():
    pools = PoolBatch.from_scores(
        np.array([[3.0, 1.0, 2.0]]),
        np.array([[1.0, 3.0, 2.0]]),
        ["z", "a", "b"],
        depth=2,
    )
    assert rank_pools(pools, np.array([0.5]), top_k=3).tolist() == [[1, 2, 0]]
    assert rank_pools(pools, np.array([1.0]), top_k=1).tolist() == [[0]]
    with pytest.raises(ValueError, match="alpha"):
        rank_pools(pools, np.array([float("nan")]))
    with pytest.raises(ValueError, match="unique"):
        PoolBatch(np.array([[0, 0]]), np.ones((1, 2, 2)), np.ones((1, 2), bool))


def test_full_qrels_idcg_counts_relevant_documents_outside_pool():
    pools = PoolBatch.from_scores(np.array([[2.0, 1.0]]), np.zeros((1, 2)), ["a", "b"])
    utilities = graded_utilities(pools, [{0: 2.0, 99: 3.0}], top_k=2)
    assert utilities.shape == (1, 101)
    assert utilities[0, 100] == pytest.approx(3 / (7 + 3 / math.log2(3)))

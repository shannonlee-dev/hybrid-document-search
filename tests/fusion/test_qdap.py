"""Exercise the QDAP-S distribution, loss, fitting and frozen selection contract."""

import math

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from evaluation.qdap import (  # noqa: E402
    FitConfig,
    PoolBatch,
    fit_qdap,
    graded_utilities,
    rank_pools,
    select_candidate,
    validate_partitions,
)
from fusion.qdap import (  # noqa: E402
    QDAPSmall,
    distribution_targets,
    qdap_loss,
    read_alpha,
)


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
    p = torch.zeros(1, 101)
    p[0, 10] = 0.5
    p[0, 90] = 0.5
    assert read_alpha(p, "mean").item() == pytest.approx(0.5)
    assert read_alpha(p, "argmax").item() == pytest.approx(0.1)
    with pytest.raises(ValueError, match="readout"):
        read_alpha(p, "median")


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


def test_fitting_returns_minimum_calibration_epoch_and_restorable_state():
    torch.set_num_threads(1)
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(11, 4)).astype("float32")
    curves = rng.uniform(size=(11, 101)).astype("float32")
    config = FitConfig(fit_size=8, calibration_size=3, epochs=2, batch_size=3)
    result = fit_qdap(vectors, curves, seed=42, config=config)
    assert (
        result.chosen_epoch
        == min(result.history, key=lambda x: x["calibration_loss"])["epoch"]
    )
    restored = QDAPSmall(4)
    restored.load_state_dict(result.state_dict)
    assert torch.isfinite(restored(torch.tensor(vectors))).all()
    repeated = fit_qdap(vectors, curves, seed=42, config=config)
    assert result.history == repeated.history
    for name, value in result.state_dict.items():
        assert torch.equal(value, repeated.state_dict[name])


def test_selection_uses_validation_only_and_stable_ties():
    records = [
        {"name": "literal-42:mean", "metrics": {"ndcg@10": 0.7}},
        {"name": "reference-43:argmax", "metrics": {"ndcg@10": 0.8}},
        {"name": "literal-44:mean", "metrics": {"ndcg@10": 0.8}},
    ]
    assert select_candidate(records)["name"] == "reference-43:argmax"
    with pytest.raises(ValueError, match="nonempty"):
        select_candidate([])


def test_partition_guard_rejects_parent_or_normalized_text_leakage():
    fit = [{"query_id": "f", "parent_doc_id": "p", "text": "A B"}]
    calibration = [{"query_id": "c", "parent_doc_id": "q", "text": "C"}]
    evaluation = [{"query_id": "v", "parent_doc_id": "r", "text": " a  b "}]
    with pytest.raises(ValueError, match="text"):
        validate_partitions(fit, calibration, evaluation)
    calibration[0]["parent_doc_id"] = "p"
    with pytest.raises(ValueError, match="parent"):
        validate_partitions(fit, calibration, [])

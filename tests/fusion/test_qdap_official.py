"""Independent encoder training and official Validation selection contracts."""

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


def test_large_copies_dense_encoder_and_updates_every_block():
    from fusion.qdap import QDAPLarge, distribution_targets, qdap_loss

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.embedding = torch.nn.Embedding(10, 4)
            self.layer = torch.nn.Linear(4, 4)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(
                last_hidden_state=self.layer(self.embedding(input_ids))
            )

    torch.manual_seed(42)
    dense = Encoder()
    saved = {k: v.detach().clone() for k, v in dense.state_dict().items()}
    predictor = QDAPLarge.from_dense_encoder(dense)
    assert predictor.encoder is not dense
    assert all(p.requires_grad for p in predictor.parameters())
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=0.01)
    p = predictor(
        input_ids=torch.tensor([[1, 2], [3, 4]]), attention_mask=torch.ones(2, 2)
    )
    torch.testing.assert_close(p.sum(1), torch.ones(2))
    qdap_loss(p, distribution_targets(torch.randn(2, 101))).backward()
    optimizer.step()
    for key, value in predictor.encoder.state_dict().items():
        assert not torch.equal(value, saved[key])
        torch.testing.assert_close(dense.state_dict()[key], saved[key])


def test_official_partition_guard_retains_text_overlap_but_rejects_parent_leakage():
    from evaluation.qdap_training import audit_official_partitions

    train = [{"query_id": "t", "parent_doc_id": "p1", "text": "same"}]
    validation = [{"query_id": "v", "parent_doc_id": "p2", "text": "same"}]
    test = [{"query_id": "e", "parent_doc_id": "p3", "text": "else"}]
    audit = audit_official_partitions(train, validation, test)
    assert audit["train_queries"] == 1
    assert audit["train_validation_text_overlap"] == 1
    with pytest.raises(ValueError, match="parent"):
        audit_official_partitions(
            train, validation, [{**test[0], "parent_doc_id": "p1"}]
        )


def test_checkpoint_selection_rejects_test_and_prefers_earliest_tie():
    from evaluation.qdap_training import select_validation_checkpoint

    rows = [
        {"epoch": 1, "readout": "mean", "metrics": {"ndcg@10": 0.5}},
        {"epoch": 2, "readout": "argmax", "metrics": {"ndcg@10": 0.5}},
    ]
    assert select_validation_checkpoint(rows, split="validation") == rows[0]
    with pytest.raises(ValueError, match="validation"):
        select_validation_checkpoint(rows, split="test")


def test_official_fitting_trains_every_row_and_freezes_validation_choice(tmp_path):
    from evaluation.qdap import PoolBatch
    from evaluation.qdap_training import fit_official
    from fusion.qdap import QDAPSmall

    torch.manual_seed(12)
    model = QDAPSmall(4)
    initial = model.linear.weight.detach().clone()
    x = torch.randn(6, 4)
    utility = np.zeros((6, 101), np.float32)
    utility[:, 60:] = 1
    seen = []

    def train(indices):
        seen.extend(indices.tolist())
        return model(x[indices])

    def validation():
        return np.full((2, 101), 1 / 101)

    pools = PoolBatch.from_scores([[1, 0], [1, 0]], [[0, 1], [0, 1]], ["a", "b"])
    report = fit_official(
        model,
        utility,
        train,
        validation,
        pools,
        [{0: 1}, {0: 1}],
        tmp_path / "fit",
        epochs=2,
        microbatch=2,
        effective_batch=4,
    )
    assert sorted(seen) == sorted(list(range(6)) * 2)
    assert report["train_queries"] == 6
    assert report["selection_split"] == "validation"
    assert report["selected"]["epoch"] == 1
    assert report["selected"]["readout"] == "mean"
    assert not torch.equal(initial, model.linear.weight)
    assert (tmp_path / "fit/selected.pt").is_file()


def test_frozen_evaluation_rejects_changed_input_before_test(tmp_path):
    import json

    from evaluation.qdap_benchmark import sha
    from evaluation.qdap_execution import frozen_selection

    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    score = tmp_path / "scores.npy"
    score.write_bytes(b"original scores")
    selection = {
        "selection_split": "validation",
        "protocol_sha256": sha(protocol),
        "selected": {},
        "source_sha256": {},
        "evaluation_input_sha256": {str(score): sha(score)},
    }
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    frozen_selection(SimpleNamespace(output=tmp_path))
    score.write_bytes(b"changed scores")
    with pytest.raises(ValueError, match="input changed"):
        frozen_selection(SimpleNamespace(output=tmp_path))

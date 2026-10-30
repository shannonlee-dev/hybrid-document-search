"""Independent encoder training and official Validation selection contracts."""

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


def test_large_trains_separate_encoder_without_changing_retrieval_encoder():
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
    retrieval_encoder = Encoder().eval().requires_grad_(False)
    saved = {k: v.detach().clone() for k, v in retrieval_encoder.state_dict().items()}
    # Load identical checkpoint weights into a separately owned prediction encoder.
    encoder = Encoder()
    encoder.load_state_dict(saved)
    encoder.requires_grad_(False)
    predictor = QDAPLarge(encoder)
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
        torch.testing.assert_close(
            retrieval_encoder.state_dict()[key], saved[key], rtol=0, atol=0
        )
    assert all(p.grad is None for p in retrieval_encoder.parameters())
    assert all(not p.requires_grad for p in retrieval_encoder.parameters())


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


@pytest.mark.parametrize("changed", [None, "protocol", "checkpoint"])
@pytest.mark.parametrize("legacy_state", ["missing", "changed"])
def test_frozen_selection_ignores_historical_dat_cache_but_checks_guards(
    tmp_path, changed, legacy_state
):
    import json

    from evaluation.qdap_benchmark import sha
    from evaluation.qdap_execution import frozen_selection

    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"original checkpoint")
    legacy = tmp_path / "artifacts/paper-guided/dat-local/bccard"
    selection = {
        "selection_split": "validation",
        "protocol_sha256": sha(protocol),
        "selected": {
            "S": {"checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint)}
        },
        "source_sha256": {},
        "evaluation_input_sha256": {
            str(legacy / name): "0" * 64
            for name in ("dense.npy", "sparse.npy", "queries.json")
        },
    }
    if legacy_state == "changed":
        legacy.mkdir(parents=True)
        for name in ("dense.npy", "sparse.npy", "queries.json"):
            (legacy / name).write_bytes(b"unused changed DAT cache")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    original = selection_path.read_bytes()
    if changed == "protocol":
        protocol.write_text('{"changed": true}')
    elif changed == "checkpoint":
        checkpoint.write_bytes(b"changed checkpoint")

    if changed is None:
        assert frozen_selection(SimpleNamespace(output=tmp_path)) == selection
    else:
        with pytest.raises(ValueError, match=f"{changed} changed"):
            frozen_selection(SimpleNamespace(output=tmp_path))
    assert selection_path.read_bytes() == original
    assert legacy.exists() == (legacy_state == "changed")


@pytest.mark.parametrize(
    "source",
    [
        "data/processed/bccard/corpus.jsonl",
        "data/processed/bccard/queries_test.jsonl",
        "data/processed/bccard/qrels_test.jsonl",
        "data/processed/bccard/manifest.json",
        "docs/experiments/adaptive/router.json",
        "artifacts/bccard-adaptive-vectors-resumed/document-vectors.npy",
        "artifacts/bccard-adaptive-vectors-resumed/query-vectors-test.npy",
        "artifacts/bccard-adaptive-vectors-resumed/encoding.json",
        "model/pytorch_model.bin",
        "model/config.json",
        "model/tokenizer.json",
        "other/dense.npy",
    ],
)
def test_frozen_selection_rejects_changed_evaluation_input(tmp_path, source):
    import json

    from evaluation.qdap_benchmark import sha
    from evaluation.qdap_execution import frozen_selection

    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    current = tmp_path / source
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_bytes(b"frozen evaluation input")
    legacy = tmp_path / "artifacts/paper-guided/dat-local/bccard"
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "selection_split": "validation",
                "protocol_sha256": sha(protocol),
                "selected": {},
                "evaluation_input_sha256": {
                    **{
                        str(legacy / name): "0" * 64
                        for name in ("dense.npy", "sparse.npy", "queries.json")
                    },
                    str(current): sha(current),
                },
            }
        )
    )
    original = selection_path.read_bytes()
    current.write_bytes(b"changed evaluation input")
    with pytest.raises(ValueError, match="evaluation input changed after selection"):
        frozen_selection(SimpleNamespace(output=tmp_path))
    assert selection_path.read_bytes() == original

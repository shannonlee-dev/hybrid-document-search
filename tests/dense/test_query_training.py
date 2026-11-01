"""Exercise query-only updates, exclusion rules and validation-only selection."""

import hashlib
import json
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from evaluation.query_encoder import (  # noqa: E402
    fixed_fusion_rankings,
    select_fusion,
)
from evaluation.query_encoder_training import (  # noqa: E402
    checkpoint_selection,
    configure_trainable_blocks,
    contrastive_candidates,
    eligible_training_indices,
    encode_queries,
    query_training_step,
    train_query_encoder,
)


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Linear(3, 3)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(2)])

    def forward(self, features):
        vector = self.embedding(features["input"])
        for block in self.blocks:
            vector = torch.tanh(block(vector))
        return {"sentence_embedding": vector}


def test_optimizer_updates_only_selected_query_block_and_never_documents():
    torch.manual_seed(35)
    model = TinyEncoder()
    selected = configure_trainable_blocks(model, model.blocks, 1)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    documents = torch.nn.functional.normalize(torch.randn(12, 3), dim=-1)
    documents.requires_grad_(True)
    document_snapshot = documents.detach().clone()
    baseline = torch.nn.functional.normalize(torch.randn(2, 3), dim=-1)
    optimizer = torch.optim.AdamW(selected.parameters(), lr=2e-5, weight_decay=0.01)
    loss = query_training_step(
        model,
        {"input": torch.randn(2, 3)},
        optimizer,
        documents,
        baseline,
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        torch.arange(12),
    )
    assert np.isfinite(loss)
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.named_parameters()
        if name.startswith("blocks.1.")
    )
    for name, value in model.named_parameters():
        if not name.startswith("blocks.1."):
            assert torch.equal(before[name], value)
            assert value.grad is None
        assert value.dtype == torch.float32
    assert torch.equal(documents, document_snapshot)
    assert documents.grad is None


def test_same_parent_hard_negatives_and_in_batch_positives_are_masked():
    documents = torch.eye(5)
    candidates, forbidden = contrastive_candidates(
        documents[:2],
        documents,
        torch.tensor([0, 1]),
        torch.tensor([0, 0]),
        torch.tensor([0, 0, 1, 2, 3]),
        hard_negatives=2,
    )
    assert candidates[:2].tolist() == [0, 1]
    assert set(candidates[2:].tolist()) <= {2, 3, 4}
    assert not forbidden[0, 0] and not forbidden[1, 1]
    assert forbidden[0, 1] and forbidden[1, 0]


def test_insufficient_eligible_negative_pool_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        contrastive_candidates(
            torch.eye(2)[:1],
            torch.eye(2),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0, 0]),
            hard_negatives=1,
        )


def test_invalid_layer_count_does_not_mutate_model():
    model = TinyEncoder()
    with pytest.raises(ValueError):
        configure_trainable_blocks(model, model.blocks, 3)
    assert all(p.requires_grad for p in model.parameters())


def test_duplicate_texts_are_excluded_and_parent_overlap_is_rejected():
    train = [
        {"text": "  SAME Text ", "parent_doc_id": "a"},
        {"text": "train", "parent_doc_id": "b"},
    ]
    heldout = [{"text": "same text", "parent_doc_id": "c"}]
    assert eligible_training_indices(train, heldout).tolist() == [1]
    with pytest.raises(ValueError, match="parent"):
        eligible_training_indices(train, [{"text": "other", "parent_doc_id": "b"}])


def test_baseline_is_eligible_and_checkpoint_ties_choose_earliest():
    assert (
        checkpoint_selection(
            [{"step": 0, "validation_ndcg": 0.8}, {"step": 64, "validation_ndcg": 0.7}]
        )["step"]
        == 0
    )
    assert (
        checkpoint_selection(
            [
                {"step": 0, "validation_ndcg": 0.7},
                {"step": 64, "validation_ndcg": 0.8},
                {"step": 128, "validation_ndcg": 0.8},
            ]
        )["step"]
        == 64
    )
    with pytest.raises(ValueError, match="baseline"):
        checkpoint_selection([{"step": 64, "validation_ndcg": 0.8}])


def test_fusion_ties_follow_corpus_order_and_empty_ranges_remain_finite():
    rankings = fixed_fusion_rankings(
        np.zeros(5), np.zeros(5), [0, 0.5, 1], pool_k=3, k=2
    )
    np.testing.assert_array_equal(rankings, [[0, 1]] * 3)


def test_fusion_selection_accepts_validation_only_and_ties_prefer_baseline():
    # 테스트 지표를 입력받지 않는 API로 선택 단계의 정보 경계를 고정한다.
    quality = np.array([[[0.7, 0, 1], [0.8, 1, 1]]])
    selection = select_fusion({"baseline": quality, "seed43": quality}, [0, 1])
    assert selection["selected_encoder"] == "baseline"
    assert selection["models"]["baseline"]["alpha_dense"] == 1
    with pytest.raises(ValueError, match="baseline"):
        select_fusion({"seed43": quality}, [0, 1])


def test_training_driver_saves_reloadable_choice_and_provenance_without_test_labels(
    tmp_path, monkeypatch
):
    import sys

    class LocalEncoder(TinyEncoder):
        def __getitem__(self, index):
            return SimpleNamespace(
                auto_model=SimpleNamespace(encoder=SimpleNamespace(layer=self.blocks))
            )

        def tokenizer(self, texts, **kwargs):
            values = [
                [1.0, 0.5, 0.2] if text == "train" else [0.2, 1.0, 0.3]
                for text in texts
            ]
            return {"input": torch.tensor(values)}

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=lambda *args, **kwargs: LocalEncoder()),
    )
    data, cache, model_dir = [tmp_path / name for name in ("data", "cache", "model")]
    for directory in (data, cache, model_dir):
        directory.mkdir()
    (model_dir / "pytorch_model.bin").write_bytes(b"local-fixture")
    docs = [
        {"document_id": f"base_bcai_{i:06}_000", "text": f"document {i}"}
        for i in range(12)
    ]
    (data / "corpus.jsonl").write_text("".join(json.dumps(row) + "\n" for row in docs))
    for index, split in enumerate(("train", "validation", "test")):
        query = {
            "query_id": split,
            "text": split,
            "provenance": "base",
            "parent_doc_id": f"base_bcai_{index:06}",
        }
        (data / f"queries_{split}.jsonl").write_text(json.dumps(query) + "\n")
        if split != "test":
            (data / f"qrels_{split}.jsonl").write_text(
                json.dumps(
                    {
                        "query_id": split,
                        "document_id": docs[index]["document_id"],
                        "relevance": 3,
                    }
                )
                + "\n"
            )

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "schema_version": "1",
        "dataset_revision": "1" * 40,
        "splits": ["train", "validation", "test"],
        "output_sha256": {p.name: digest(p) for p in data.glob("*.jsonl")},
        "output_counts": {
            p.name: len(p.read_text().splitlines()) for p in data.glob("*.jsonl")
        },
    }
    (data / "manifest.json").write_text(json.dumps(manifest))
    vectors = np.random.default_rng(8).normal(size=(12, 3)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(cache / "document-vectors.npy", vectors)
    np.save(cache / "query-vectors-train.npy", vectors[:1])
    encoding = {
        "model": {"revision": "2" * 40},
        "dataset_manifest_sha256": digest(data / "manifest.json"),
        "document_ids_sha256": hashlib.sha256(
            "\n".join(row["document_id"] for row in docs).encode()
        ).hexdigest(),
        "vector_sha256": {p.name: digest(p) for p in cache.glob("*.npy")},
        "splits": {
            split: {"query_ids": [split]} for split in ("train", "validation", "test")
        },
    }
    (cache / "encoding.json").write_text(json.dumps(encoding))
    run = tmp_path / "tiny-seed43"
    frozen = train_query_encoder(
        data,
        cache,
        model_dir,
        run,
        seed=43,
        steps=2,
        layers=1,
        batch_size=1,
        device="cpu",
    )
    protocol = json.loads((run / "protocol.json").read_text())
    assert protocol["run_id"] == "tiny-seed43"
    assert protocol["steps"] == 2
    assert protocol["source_sha256"]["evaluation/query_encoder_training.py"]
    assert frozen["protocol_sha256"] == digest(run / "protocol.json")
    assert [row["step"] for row in frozen["history"]] == [0, 2]
    assert frozen["checkpoint_sha256"] == digest(run / "selected-last-block.pt")
    torch.manual_seed(43)
    reloaded = LocalEncoder()
    selected = configure_trainable_blocks(reloaded, reloaded.blocks, 1)
    selected.load_state_dict(
        torch.load(run / "selected-last-block.pt", weights_only=True)
    )
    np.testing.assert_array_equal(
        encode_queries(reloaded, ["validation"]),
        np.load(run / "selected-validation-vectors.npy"),
    )

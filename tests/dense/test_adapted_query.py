"""Adapted query embedder: checkpoint identity, block loading and document refusal."""

import hashlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from retrievers.adapted_query import AdaptedQueryEmbedder  # noqa: E402
from retrievers.dense import DenseConfig  # noqa: E402


class _FakeModel:
    """Mimic SentenceTransformer: model[0].auto_model.encoder.layer and encode()."""

    def __init__(self, path, device=None, local_files_only=False):
        assert local_files_only
        torch.manual_seed(0)
        layer = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(6))
        self.auto_model = types.SimpleNamespace(
            encoder=types.SimpleNamespace(layer=layer)
        )
        self.calls = []

    def __getitem__(self, index):
        return self

    def eval(self):
        return self

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        weight = self.auto_model.encoder.layer[-1].weight.detach().numpy()
        vector = np.array([weight[0, 0], 1.0], dtype=np.float32)
        return np.repeat((vector / np.linalg.norm(vector))[None, :], len(texts), axis=0)


@pytest.fixture
def fake_st(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def _checkpoint(tmp_path, layers=4, value=0.5):
    blocks = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(layers))
    for block in blocks:
        torch.nn.init.constant_(block.weight, value)
    state = blocks.state_dict() if layers > 1 else blocks[0].state_dict()
    path = tmp_path / "selected-last-block.pt"
    torch.save(state, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _embedder(tmp_path, path, digest, layers=4):
    (tmp_path / "base").mkdir(exist_ok=True)
    return AdaptedQueryEmbedder(
        DenseConfig(model_name="fixture"),
        base_model_path=tmp_path / "base",
        checkpoint_path=path,
        checkpoint_sha256=digest,
        layers=layers,
    )


def test_checkpoint_replaces_only_the_last_blocks(tmp_path, fake_st):
    path, digest = _checkpoint(tmp_path)
    embedder = _embedder(tmp_path, path, digest)
    model = embedder.load_model()
    layers = model.auto_model.encoder.layer
    assert all(torch.all(block.weight == 0.5) for block in layers[-4:])
    assert not any(torch.all(block.weight == 0.5) for block in layers[:2])
    vector = embedder.encode_query("카드")
    assert vector.shape == (1, 2)
    texts, kwargs = model.calls[-1]
    assert (
        texts == ["카드"] and kwargs["prompt"] == "" and kwargs["normalize_embeddings"]
    )


def test_single_block_checkpoint_uses_bare_block_state(tmp_path, fake_st):
    path, digest = _checkpoint(tmp_path, layers=1, value=0.25)
    model = _embedder(tmp_path, path, digest, layers=1).load_model()
    assert torch.all(model.auto_model.encoder.layer[-1].weight == 0.25)


def test_wrong_checkpoint_hash_is_rejected(tmp_path):
    path, _ = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        _embedder(tmp_path, path, "0" * 64)


def test_mismatched_block_count_fails_strict_load(tmp_path, fake_st):
    path, digest = _checkpoint(tmp_path, layers=3)
    with pytest.raises(RuntimeError):
        _embedder(tmp_path, path, digest, layers=4).load_model()


def test_documents_are_never_encoded_by_the_adapted_model(tmp_path):
    path, digest = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="must not encode documents"):
        _embedder(tmp_path, path, digest).encode_documents(["문서"])


def test_missing_base_model_is_rejected(tmp_path):
    path, digest = _checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError):
        AdaptedQueryEmbedder(
            DenseConfig(model_name="fixture"),
            base_model_path=tmp_path / "missing",
            checkpoint_path=path,
            checkpoint_sha256=digest,
        )

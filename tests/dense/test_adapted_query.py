"""Adapted query embedder: fail-closed identity, block loading and document refusal."""

import hashlib
import json
import sys
import types

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from retrievers.adapted_query import (  # noqa: E402
    AdaptedQueryEmbedder,
    load_checkpoint_identity,
    snapshot_files_sha256,
)
from retrievers.dense import DenseConfig  # noqa: E402

MODEL = DenseConfig(model_name="BAAI/bge-m3")  # revision pinned by the registry


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
        return np.repeat((vector / np.linalg.norm(vector))[None, :], len(texts), 0)


@pytest.fixture
def fake_st(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def _setup(tmp_path, layers=4, value=0.5):
    """Create a fake base snapshot, checkpoint and matching identity dict."""
    base = tmp_path / "base"
    (base / "1_Pooling").mkdir(parents=True, exist_ok=True)
    (base / "pytorch_model.bin").write_bytes(b"weights")
    (base / "config.json").write_text("{}")
    (base / "1_Pooling" / "config.json").write_text('{"pooling": "cls"}')
    (base / "README.md").write_text("not hashed")
    blocks = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(layers))
    for block in blocks:
        torch.nn.init.constant_(block.weight, value)
    state = blocks.state_dict() if layers > 1 else blocks[0].state_dict()
    checkpoint = tmp_path / "selected-last-block.pt"
    torch.save(state, checkpoint)
    identity = {
        "schema_version": 1,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "layers": layers,
        "base_model": {
            "model_name": MODEL.model_name,
            "revision": MODEL.revision,
            "files_sha256": snapshot_files_sha256(base),
        },
    }
    return base, checkpoint, identity


def _embedder(base, checkpoint, identity, config=MODEL):
    return AdaptedQueryEmbedder(
        config, base_model_path=base, checkpoint_path=checkpoint, identity=identity
    )


def test_matching_identity_loads_and_replaces_only_the_last_blocks(tmp_path, fake_st):
    embedder = _embedder(*_setup(tmp_path))
    model = embedder.load_model()
    layers = model.auto_model.encoder.layer
    assert all(torch.all(block.weight == 0.5) for block in layers[-4:])
    assert not any(torch.all(block.weight == 0.5) for block in layers[:2])
    assert embedder.encode_query("카드").shape == (1, 2)
    texts, kwargs = model.calls[-1]
    assert texts == ["카드"] and kwargs["prompt"] == ""


def test_single_block_checkpoint_uses_bare_block_state(tmp_path, fake_st):
    model = _embedder(*_setup(tmp_path, layers=1, value=0.25)).load_model()
    assert torch.all(model.auto_model.encoder.layer[-1].weight == 0.25)


def test_checkpoint_hash_mismatch_fails_closed(tmp_path):
    base, checkpoint, identity = _setup(tmp_path)
    identity["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        _embedder(base, checkpoint, identity)


@pytest.mark.parametrize("change", ["modify", "extra", "missing"])
def test_base_snapshot_mismatch_fails_closed(tmp_path, change):
    base, checkpoint, identity = _setup(tmp_path)
    if change == "modify":
        (base / "pytorch_model.bin").write_bytes(b"other weights")
    elif change == "extra":
        (base / "model.safetensors").write_bytes(b"unverified")
    else:
        (base / "config.json").unlink()
    with pytest.raises(ValueError, match="base snapshot"):
        _embedder(base, checkpoint, identity)


@pytest.mark.parametrize(
    "config",
    [
        DenseConfig(model_name="BAAI/bge-m3", revision="1" * 40),
        DenseConfig(model_name="intfloat/multilingual-e5-base"),
        DenseConfig(model_name="cached-vectors"),
    ],
)
def test_index_model_mismatch_fails_closed(tmp_path, config):
    with pytest.raises(ValueError, match="index embedding model"):
        _embedder(*_setup(tmp_path), config=config)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda i: i.pop("base_model"),
        lambda i: i["base_model"].pop("files_sha256"),
        lambda i: i["base_model"].update(files_sha256={"config.json": "a" * 64}),
        lambda i: i.update(schema_version=2),
        lambda i: i.update(layers=0),
        lambda i: i.update(checkpoint_sha256="xyz"),
    ],
)
def test_incomplete_identity_file_is_rejected(tmp_path, mutate):
    _, _, identity = _setup(tmp_path)
    mutate(identity)
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint_identity(path)


def test_missing_identity_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint_identity(tmp_path / "missing.json")


def test_documents_are_never_encoded_by_the_adapted_model(tmp_path):
    with pytest.raises(ValueError, match="must not encode documents"):
        _embedder(*_setup(tmp_path)).encode_documents(["문서"])


def test_snapshot_rule_matches_training_record(tmp_path):
    pytest.importorskip("bm25s")
    from evaluation.query_encoder_report import model_identity

    base, _, _ = _setup(tmp_path)
    assert snapshot_files_sha256(base) == model_identity(base)

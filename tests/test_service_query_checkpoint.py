"""Opt-in query checkpoint in the search service; default behavior is unchanged."""

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("faiss")
pytest.importorskip("bm25s")
pytest.importorskip("sklearn")

from app.api import _query_checkpoint_from_env  # noqa: E402
from app.schemas import RetrievalMethod  # noqa: E402
from app.service import MethodState, QueryCheckpoint, build_search_service  # noqa: E402
from data.loader import Document  # noqa: E402
from indexing.faiss_index import FaissIndex  # noqa: E402
from retrievers.adapted_query import snapshot_files_sha256  # noqa: E402
from retrievers.dense import DenseConfig, DenseRetriever  # noqa: E402

ENV = (
    "DENSE_QUERY_CHECKPOINT",
    "DENSE_QUERY_CHECKPOINT_IDENTITY",
    "DENSE_QUERY_BASE_MODEL",
)
INDEX_MODEL = DenseConfig(model_name="BAAI/bge-m3")


class _FakeModel:
    def __init__(self, path, device=None, local_files_only=False, revision=None):
        layer = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(6))
        self.auto_model = types.SimpleNamespace(
            encoder=types.SimpleNamespace(layer=layer)
        )

    def __getitem__(self, index):
        return self

    def eval(self):
        return self

    def encode(self, texts, **kwargs):
        return np.tile(np.array([[0.6, 0.8]], dtype=np.float32), (len(texts), 1))


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    documents = [Document("a", "카드 해지"), Document("b", "연회비 환불")]
    (tmp_path / "corpus.jsonl").write_text(
        "".join(
            json.dumps(
                {"document_id": d.document_id, "text": d.text, "title": None},
                ensure_ascii=False,
            )
            + "\n"
            for d in documents
        ),
        encoding="utf-8",
    )
    DenseRetriever(
        index=FaissIndex.build([[1.0, 0.0], [0.0, 1.0]]),
        documents=documents,
        embedder=types.SimpleNamespace(config=INDEX_MODEL),
    ).save(tmp_path / "index")
    blocks = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(4))
    torch.save(blocks.state_dict(), tmp_path / "ckpt.pt")
    base = tmp_path / "base"
    base.mkdir()
    (base / "pytorch_model.bin").write_bytes(b"weights")
    (base / "config.json").write_text("{}")
    identity = {
        "schema_version": 1,
        "checkpoint_sha256": hashlib.sha256(
            (tmp_path / "ckpt.pt").read_bytes()
        ).hexdigest(),
        "layers": 4,
        "base_model": {
            "model_name": INDEX_MODEL.model_name,
            "revision": INDEX_MODEL.revision,
            "files_sha256": snapshot_files_sha256(base),
        },
    }
    (tmp_path / "identity.json").write_text(json.dumps(identity))
    return tmp_path, identity


def _build(root: Path, identity_name="identity.json"):
    return build_search_service(
        root / "corpus.jsonl",
        dense_index_path=root / "index",
        dense_query_checkpoint=QueryCheckpoint(
            root / "base", root / "ckpt.pt", root / identity_name
        ),
    )


def _assert_fail_closed(service):
    status = service.availability()
    assert status[RetrievalMethod.DENSE].state is MethodState.LOAD_FAILED
    assert status[RetrievalMethod.HYBRID].state is MethodState.UPSTREAM_UNAVAILABLE
    assert status[RetrievalMethod.BM25].available
    assert service.retriever(RetrievalMethod.DENSE) is None


def test_default_service_keeps_the_index_embedder(corpus):
    root, _ = corpus
    service = build_search_service(
        root / "corpus.jsonl", dense_index_path=root / "index"
    )
    dense = service.retriever(RetrievalMethod.DENSE)
    assert type(dense.embedder).__name__ == "DenseEmbedder"
    assert service.retriever(RetrievalMethod.HYBRID) is not None


def test_matching_identity_replaces_only_the_query_encoder(corpus):
    root, _ = corpus
    service = _build(root)
    dense = service.retriever(RetrievalMethod.DENSE)
    assert type(dense.embedder).__name__ == "AdaptedQueryEmbedder"
    np.testing.assert_array_equal(dense.index.snapshot_vectors(), [[1, 0], [0, 1]])
    assert service.retriever(RetrievalMethod.HYBRID)._embedder is dense.embedder
    hits = service.search("카드", RetrievalMethod.DENSE, 2)
    assert [r.document_id for r in hits] == ["b", "a"]


def test_missing_identity_file_fails_closed(corpus):
    root, _ = corpus
    _assert_fail_closed(_build(root, "absent.json"))


def test_identity_mismatch_fails_closed(corpus):
    root, identity = corpus
    identity["base_model"]["revision"] = "1" * 40
    (root / "other.json").write_text(json.dumps(identity))
    service = _build(root, "other.json")
    _assert_fail_closed(service)
    assert "base" not in (service.availability()[RetrievalMethod.DENSE].reason or "")


def test_tampered_base_snapshot_fails_closed(corpus):
    root, _ = corpus
    (root / "base" / "pytorch_model.bin").write_bytes(b"swapped")
    _assert_fail_closed(_build(root))


def test_env_parsing(monkeypatch, tmp_path):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    assert _query_checkpoint_from_env() is None
    monkeypatch.setenv("DENSE_QUERY_CHECKPOINT", str(tmp_path / "c.pt"))
    with pytest.raises(ValueError, match="set together"):
        _query_checkpoint_from_env()
    monkeypatch.setenv("DENSE_QUERY_CHECKPOINT_IDENTITY", str(tmp_path / "i.json"))
    monkeypatch.setenv("DENSE_QUERY_BASE_MODEL", str(tmp_path))
    assert _query_checkpoint_from_env() == QueryCheckpoint(
        tmp_path, tmp_path / "c.pt", tmp_path / "i.json"
    )


def test_identity_builder_uses_recorded_run_evidence(tmp_path):
    from scripts.make_query_checkpoint_identity import build_identity

    run = tmp_path / "train"
    run.mkdir()
    (run / "selected-last-block.pt").write_bytes(b"checkpoint")
    protocol = {
        "revision": INDEX_MODEL.revision,
        "trainable_layers": 4,
        "model_files_sha256": {"pytorch_model.bin": "a" * 64},
    }
    (run / "protocol.json").write_text(json.dumps(protocol))
    digest = hashlib.sha256((run / "protocol.json").read_bytes()).hexdigest()
    frozen = {
        "run_id": "train",
        "selected_step": 4096,
        "protocol_sha256": digest,
        "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
    }
    (run / "frozen.json").write_text(json.dumps(frozen))
    encoding = tmp_path / "encoding.json"
    encoding.write_text(
        json.dumps(
            {"model": {"model_name": "BAAI/bge-m3", "revision": protocol["revision"]}}
        )
    )
    identity = build_identity(run, encoding)
    assert identity["base_model"]["files_sha256"] == protocol["model_files_sha256"]
    assert identity["checkpoint_sha256"] == frozen["checkpoint_sha256"]
    (run / "selected-last-block.pt").write_bytes(b"swapped")
    with pytest.raises(ValueError, match="checkpoint"):
        build_identity(run, encoding)

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
from retrievers.dense import DenseConfig, DenseRetriever  # noqa: E402

ENV = (
    "DENSE_QUERY_CHECKPOINT",
    "DENSE_QUERY_CHECKPOINT_SHA256",
    "DENSE_QUERY_BASE_MODEL",
)


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
    path = tmp_path / "corpus.jsonl"
    path.write_text(
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
        embedder=types.SimpleNamespace(config=DenseConfig(model_name="fixture")),
    ).save(tmp_path / "index")
    blocks = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(4))
    torch.save(blocks.state_dict(), tmp_path / "ckpt.pt")
    (tmp_path / "base").mkdir()
    digest = hashlib.sha256((tmp_path / "ckpt.pt").read_bytes()).hexdigest()
    return tmp_path, digest


def _checkpoint(root: Path, digest: str) -> QueryCheckpoint:
    return QueryCheckpoint(root / "base", root / "ckpt.pt", digest)


def test_default_service_keeps_the_index_embedder(corpus):
    root, _ = corpus
    service = build_search_service(
        root / "corpus.jsonl", dense_index_path=root / "index"
    )
    dense = service.retriever(RetrievalMethod.DENSE)
    assert type(dense.embedder).__name__ == "DenseEmbedder"
    assert service.retriever(RetrievalMethod.HYBRID) is not None


def test_query_checkpoint_replaces_only_the_query_encoder(corpus):
    root, digest = corpus
    service = build_search_service(
        root / "corpus.jsonl",
        dense_index_path=root / "index",
        dense_query_checkpoint=_checkpoint(root, digest),
    )
    dense = service.retriever(RetrievalMethod.DENSE)
    assert type(dense.embedder).__name__ == "AdaptedQueryEmbedder"
    np.testing.assert_array_equal(dense.index.snapshot_vectors(), [[1, 0], [0, 1]])
    hybrid = service.retriever(RetrievalMethod.HYBRID)
    assert hybrid._embedder is dense.embedder
    assert [
        r.document_id for r in service.search("카드", RetrievalMethod.DENSE, 2)
    ] == ["b", "a"]


def test_bad_checkpoint_disables_dense_and_hybrid_but_not_bm25(corpus):
    root, _ = corpus
    service = build_search_service(
        root / "corpus.jsonl",
        dense_index_path=root / "index",
        dense_query_checkpoint=_checkpoint(root, "0" * 64),
    )
    status = service.availability()
    assert status[RetrievalMethod.DENSE].state is MethodState.LOAD_FAILED
    assert status[RetrievalMethod.HYBRID].state is MethodState.UPSTREAM_UNAVAILABLE
    assert status[RetrievalMethod.BM25].available
    assert "ckpt" not in (status[RetrievalMethod.DENSE].reason or "")


def test_env_parsing(monkeypatch, tmp_path):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    assert _query_checkpoint_from_env() is None
    monkeypatch.setenv("DENSE_QUERY_CHECKPOINT", str(tmp_path / "c.pt"))
    with pytest.raises(ValueError, match="set together"):
        _query_checkpoint_from_env()
    monkeypatch.setenv("DENSE_QUERY_CHECKPOINT_SHA256", "a" * 64)
    monkeypatch.setenv("DENSE_QUERY_BASE_MODEL", str(tmp_path))
    checkpoint = _query_checkpoint_from_env()
    assert checkpoint == QueryCheckpoint(tmp_path, tmp_path / "c.pt", "a" * 64)

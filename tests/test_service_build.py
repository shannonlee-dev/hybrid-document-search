"""Verify service initialization and Hybrid wiring with isolated retriever doubles."""

import builtins
import sys
import tomllib
import types
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.schemas import RetrievalMethod
from app.service import MethodState, _is_placeholder, build_search_service
from data.loader import Document, load_prepared_documents
from retrievers.base import SearchResult

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "ko_miracl_prepared_corpus.jsonl"


class _FakeTfidf:
    def __init__(self, documents):
        self.documents = tuple(documents)

    def search(self, query, top_k):
        return _ranked(self.documents, top_k, 1.0)


class _FakeBM25:
    """Track initialization as a retriever with an implemented constructor."""

    built_from = []

    def __init__(self, documents):
        self.documents = tuple(documents)
        _FakeBM25.built_from.append(len(self.documents))

    def search(self, query, top_k):
        return _ranked(self.documents, top_k, 2.0)


class _PlaceholderBM25:
    """Represent an unimplemented retriever by omitting its constructor."""

    def search(self, query, top_k):
        raise NotImplementedError


class _HideSklearn:
    """Make ``import sklearn`` fail the way a missing package does."""

    def find_spec(self, name, path=None, target=None):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ModuleNotFoundError(f"No module named {name!r}", name="sklearn")
        return None


@pytest.fixture
def fake_tfidf_module(monkeypatch):
    # scikit-learn 없이 실행하도록 모듈을 대역으로 교체한다.
    module = types.ModuleType("retrievers.tfidf")
    module.TfidfRetriever = _FakeTfidf
    monkeypatch.setitem(sys.modules, "retrievers.tfidf", module)


@pytest.fixture
def fake_bm25(monkeypatch):
    # 선택적 BM25 의존성을 로드하지 않도록 모듈을 대역으로 교체한다.
    _FakeBM25.built_from.clear()
    module = types.ModuleType("retrievers.bm25")
    module.BM25Retriever = _FakeBM25
    monkeypatch.setitem(sys.modules, "retrievers.bm25", module)
    return _FakeBM25


def test_corpus_missing_marks_every_method_and_hides_path(tmp_path):
    missing = tmp_path / "private-dir" / "absent.jsonl"

    service = build_search_service(missing)

    statuses = service.availability()
    assert all(
        status.state is MethodState.CORPUS_UNAVAILABLE for status in statuses.values()
    )
    reason = statuses[RetrievalMethod.TFIDF].reason
    assert "prepare_dataset" in reason
    assert str(tmp_path) not in reason


def test_invalid_corpus_marks_every_method_without_raising(tmp_path):
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"document_id": "a", "text": "  "}\n', encoding="utf-8")

    service = build_search_service(broken)

    assert all(
        status.state is MethodState.CORPUS_UNAVAILABLE
        for status in service.availability().values()
    )


def test_tfidf_registered_and_dense_not_configured_by_default(fake_tfidf_module):
    service = build_search_service(FIXTURE_CORPUS)

    statuses = service.availability()
    assert statuses[RetrievalMethod.TFIDF].available
    assert statuses[RetrievalMethod.DENSE].state is MethodState.NOT_CONFIGURED
    assert "DENSE_INDEX_PATH" in statuses[RetrievalMethod.DENSE].reason


def test_missing_sklearn_reports_dependency_state(monkeypatch):
    # 이전 import가 누락된 의존성을 가리지 않도록 캐시된 모듈도 제거한다.
    for name in list(sys.modules):
        if (
            name == "sklearn"
            or name.startswith("sklearn.")
            or name == "retrievers.tfidf"
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_HideSklearn(), *sys.meta_path])

    service = build_search_service(FIXTURE_CORPUS)

    status = service.availability()[RetrievalMethod.TFIDF]
    assert status.state is MethodState.DEPENDENCY_MISSING
    assert "uv sync --extra sparse" in status.reason


def test_is_placeholder_detects_a_missing_constructor():
    class Placeholder:
        def search(self, query, top_k):
            raise NotImplementedError

    class Implemented:
        def __init__(self, documents):
            self.documents = documents

        def search(self, query, top_k):
            return []

    assert _is_placeholder(Placeholder)
    assert not _is_placeholder(Implemented)


def test_bm25_placeholder_reports_not_implemented(monkeypatch, fake_tfidf_module):
    _install_placeholder_bm25(monkeypatch)

    service = build_search_service(FIXTURE_CORPUS)

    assert (
        service.availability()[RetrievalMethod.BM25].state
        is MethodState.NOT_IMPLEMENTED
    )


def test_bm25_registers_as_available_with_the_real_implementation(fake_tfidf_module):
    pytest.importorskip("bm25s")
    pytest.importorskip("sklearn")
    service = build_search_service(FIXTURE_CORPUS)

    assert service.availability()[RetrievalMethod.BM25].available
    hits = service.search("서울", RetrievalMethod.BM25, 3)
    assert hits[0].document_id == "fixture-001#0"


@pytest.mark.parametrize(
    "import_error",
    [
        FileNotFoundError("C:/secret/dense_models.toml"),
        PermissionError("C:/secret/dense_models.toml"),
        ValueError("invalid model in C:/secret/dense_models.toml"),
        tomllib.TOMLDecodeError("invalid TOML in C:/secret/dense_models.toml"),
    ],
    ids=["missing", "unreadable", "invalid-model", "invalid-toml"],
)
def test_dense_config_import_failure_keeps_sparse_search_available(
    monkeypatch, fake_tfidf_module, fake_bm25, tmp_path, import_error
):
    original_import = builtins.__import__

    def _failing_dense_import(name, *args, **kwargs):
        if name == "retrievers.dense":
            raise import_error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _failing_dense_import)

    service = build_search_service(FIXTURE_CORPUS, dense_index_path=tmp_path)

    statuses = service.availability()
    assert statuses[RetrievalMethod.DENSE].state is MethodState.LOAD_FAILED
    assert "C:/secret" not in statuses[RetrievalMethod.DENSE].reason
    assert statuses[RetrievalMethod.HYBRID].state is MethodState.UPSTREAM_UNAVAILABLE
    for method in (RetrievalMethod.TFIDF, RetrievalMethod.BM25):
        assert service.search("서울", method, 1)


def test_dense_index_missing_is_reported_with_build_hint(
    monkeypatch, fake_tfidf_module, tmp_path
):
    _install_dense(monkeypatch, _fake_dense_class(load_error=FileNotFoundError()))

    service = build_search_service(FIXTURE_CORPUS, dense_index_path=tmp_path / "idx")

    status = service.availability()[RetrievalMethod.DENSE]
    assert status.state is MethodState.INDEX_MISSING
    assert "scripts.build_index" in status.reason
    assert str(tmp_path) not in status.reason


def test_dense_unusable_path_is_load_failed_not_a_startup_crash(
    monkeypatch, fake_tfidf_module, tmp_path
):
    # 인덱스 경로를 읽을 수 없어도 API는 계속 실행되어야 한다.
    _install_dense(monkeypatch, _fake_dense_class(load_error=NotADirectoryError()))

    service = build_search_service(FIXTURE_CORPUS, dense_index_path=tmp_path)

    assert (
        service.availability()[RetrievalMethod.DENSE].state is MethodState.LOAD_FAILED
    )


def test_dense_dependency_missing_is_reported(monkeypatch, fake_tfidf_module, tmp_path):
    _install_dense(
        monkeypatch, _fake_dense_class(load_error=ModuleNotFoundError("faiss"))
    )

    service = build_search_service(FIXTURE_CORPUS, dense_index_path=tmp_path)

    assert (
        service.availability()[RetrievalMethod.DENSE].state
        is MethodState.DEPENDENCY_MISSING
    )


def test_dense_invalid_index_is_load_failed_without_internal_detail(
    monkeypatch, fake_tfidf_module, tmp_path
):
    _install_dense(
        monkeypatch, _fake_dense_class(load_error=ValueError("checksum C:/secret"))
    )

    service = build_search_service(FIXTURE_CORPUS, dense_index_path=tmp_path)

    status = service.availability()[RetrievalMethod.DENSE]
    assert status.state is MethodState.LOAD_FAILED
    assert "C:/secret" not in status.reason


def test_dense_index_from_another_corpus_is_rejected(
    monkeypatch, fake_tfidf_module, tmp_path
):
    other = [Document("other-001#0", "다른 문서입니다.", None)]
    _install_dense(monkeypatch, _fake_dense_class(documents_override=other))

    service = build_search_service(FIXTURE_CORPUS, dense_index_path=tmp_path)

    status = service.availability()[RetrievalMethod.DENSE]
    assert status.state is MethodState.LOAD_FAILED
    assert "different corpus" in status.reason


def test_model_load_failure_is_reported_at_startup_not_per_request(
    monkeypatch, fake_tfidf_module, tmp_path
):
    _install_dense(
        monkeypatch, _fake_dense_class(warm_error=RuntimeError("model missing"))
    )

    service = build_search_service(FIXTURE_CORPUS, dense_index_path=tmp_path)

    status = service.availability()[RetrievalMethod.DENSE]
    assert status.state is MethodState.LOAD_FAILED
    assert "model missing" not in status.reason


def test_dense_loads_once_with_configured_device_and_warms_model(
    monkeypatch, fake_tfidf_module, tmp_path
):
    dense = _install_dense(monkeypatch, _fake_dense_class())

    service = build_search_service(
        FIXTURE_CORPUS, dense_index_path=tmp_path, dense_device="cpu"
    )

    assert dense.loads == [(tmp_path, "cpu")]
    assert dense.encode_queries == ["warm-up"]
    assert service.availability()[RetrievalMethod.DENSE].available


def test_hybrid_is_real_score_fusion_when_both_available(monkeypatch, tmp_path):
    pytest.importorskip("faiss")
    pytest.importorskip("bm25s")
    pytest.importorskip("sklearn")
    from fusion.score import ScoreFusionRetriever

    documents = _corpus_documents()
    bm25, dense = _real_fusion_sources(documents)
    _install_real_dense(monkeypatch, dense)

    service = build_search_service(FIXTURE_CORPUS, dense_index_path=tmp_path)

    assert service.availability()[RetrievalMethod.HYBRID].available
    hybrid = service.search("서울", RetrievalMethod.HYBRID, 4)
    # 독립적으로 만든 ScoreFusionRetriever와 같은 결과를 내야 한다.
    reference = ScoreFusionRetriever(bm25, dense).search("서울", 4)
    assert hybrid == reference
    assert hybrid[0].document_id == "fixture-001#0"


def test_hybrid_does_not_substitute_tfidf_for_missing_bm25(
    monkeypatch, fake_tfidf_module, tmp_path
):
    _install_placeholder_bm25(monkeypatch)
    _install_dense(monkeypatch, _fake_dense_class())

    service = build_search_service(FIXTURE_CORPUS, dense_index_path=tmp_path)

    status = service.availability()[RetrievalMethod.HYBRID]
    assert status.state is MethodState.UPSTREAM_UNAVAILABLE
    assert "BM25" in status.reason
    assert "TF-IDF" not in status.reason


@pytest.mark.parametrize("failure", ["normalization_disabled", "non_unit_vectors"])
def test_hybrid_validation_failure_preserves_other_api_methods(
    monkeypatch, tmp_path, caplog, failure
):
    pytest.importorskip("faiss")
    pytest.importorskip("bm25s")
    pytest.importorskip("sklearn")
    from indexing.faiss_index import FaissIndex

    _, dense = _real_fusion_sources(_corpus_documents())
    if failure == "normalization_disabled":
        dense.config = replace(dense.config, normalize_embeddings=False)
        dense.embedder.config = dense.config
    else:
        dense.index = FaissIndex.build(dense.index.snapshot_vectors() * 2)
    _install_real_dense(monkeypatch, dense)
    monkeypatch.setenv("SEARCH_CORPUS_PATH", str(FIXTURE_CORPUS))
    monkeypatch.setenv("DENSE_INDEX_PATH", str(tmp_path))

    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        methods = {
            row["method"]: row
            for row in client.get("/search/methods").json()["methods"]
        }
        assert methods["hybrid"]["state"] == MethodState.LOAD_FAILED
        assert methods["hybrid"]["available"] is False
        for method in ("tfidf", "bm25", "dense"):
            assert methods[method]["available"] is True
            response = client.post(
                "/search", json={"query": "서울", "method": method, "top_k": 2}
            )
            assert response.status_code == 200
            assert response.json()["results"]
        unavailable = client.post(
            "/search", json={"query": "서울", "method": "hybrid", "top_k": 2}
        )
        assert unavailable.status_code == 503
        assert str(tmp_path) not in unavailable.text
        assert "normalized" not in unavailable.text
    assert "Hybrid could not be initialized" in caplog.text


def test_corpus_loaded_once_and_no_index_built_per_request(monkeypatch, tmp_path):
    pytest.importorskip("faiss")
    pytest.importorskip("bm25s")
    pytest.importorskip("sklearn")
    import data.loader as loader
    from retrievers.bm25 import BM25Retriever

    calls = {"corpus": 0}
    real_load = loader.load_prepared_documents

    def _counting_load(path):
        calls["corpus"] += 1
        return real_load(path)

    monkeypatch.setattr("app.service.load_prepared_documents", _counting_load)

    bm25_builds = []

    class _TrackingBM25(BM25Retriever):
        def __init__(self, documents):
            super().__init__(documents)
            bm25_builds.append(len(self.documents))

    monkeypatch.setattr("retrievers.bm25.BM25Retriever", _TrackingBM25)

    _, dense = _real_fusion_sources(_corpus_documents())
    loads = []
    _install_real_dense(monkeypatch, dense, loads=loads)

    service = build_search_service(FIXTURE_CORPUS, dense_index_path=tmp_path)
    builds_after_startup = len(bm25_builds)
    loads_after_startup = len(loads)

    for _ in range(3):
        service.search("서울", RetrievalMethod.HYBRID, 2)
        service.search("서울", RetrievalMethod.DENSE, 2)

    assert calls["corpus"] == 1
    assert len(loads) == loads_after_startup
    assert len(bm25_builds) == builds_after_startup
    # 시작 시 예열 한 번과 질의 임베딩 여섯 번만 수행하고 corpus는 재임베딩하지 않는다.
    assert dense.embedder.calls[0] == "warm-up"
    assert len(dense.embedder.calls) == 1 + 6


def _fake_dense_class(*, load_error=None, warm_error=None, documents_override=None):
    """Build a DenseRetriever double with the attributes the service reads.

    Includes the ``config``/``index`` shape ScoreFusionRetriever needs so that
    whichever real BM25Retriever the service also builds can be fused with it.
    """
    import numpy as np

    class _FakeConfig:
        normalize_embeddings = True

    class _FakeIndex:
        def __init__(self, count):
            self._vectors = np.eye(max(count, 1), dtype=np.float32)[:count]

        def snapshot_vectors(self):
            return self._vectors

    class _FakeDense:
        loads = []
        encode_queries = []

        def __init__(self, documents):
            self.documents = tuple(documents)
            self.embedder = self
            self.config = _FakeConfig()
            self.index = _FakeIndex(len(self.documents))

        def encode_query(self, text):
            if warm_error is not None:
                raise warm_error
            _FakeDense.encode_queries.append(text)
            return None

        def search(self, query, top_k):
            self.embedder.encode_query(query)
            return _ranked(self.documents, top_k, 3.0)

        @classmethod
        def load(cls, directory, *, device=None):
            cls.loads.append((Path(directory), device))
            if load_error is not None:
                raise load_error
            documents = documents_override or _corpus_documents()
            return cls(documents)

    return _FakeDense


def _corpus_documents():
    return load_prepared_documents(FIXTURE_CORPUS)


def _ranked(documents, top_k, score_base):
    return [
        SearchResult(
            document_id=doc.document_id,
            rank=rank,
            score=score_base / rank,
            title=doc.title,
            snippet=doc.text[:200],
        )
        for rank, doc in enumerate(documents[:top_k], start=1)
    ]


def _install_dense(monkeypatch, cls):
    import retrievers.dense as dense_module

    monkeypatch.setattr(dense_module, "DenseRetriever", cls)
    return cls


def _install_placeholder_bm25(monkeypatch):
    module = types.ModuleType("retrievers.bm25")
    module.BM25Retriever = _PlaceholderBM25
    monkeypatch.setitem(sys.modules, "retrievers.bm25", module)


def _real_fusion_sources(documents):
    """Build a real BM25Retriever + DenseRetriever pair ScoreFusionRetriever accepts."""
    import numpy as np

    from indexing.faiss_index import FaissIndex
    from retrievers.bm25 import BM25Retriever
    from retrievers.dense import DenseConfig, DenseRetriever

    dim = len(documents)

    class _FixedEmbedder:
        config = DenseConfig(model_name="fixture")

        def __init__(self):
            self.calls = []

        def encode_query(self, query):
            self.calls.append(query)
            vector = np.zeros((1, dim), dtype=np.float32)
            vector[0, 0] = 1.0
            return vector

    dense = DenseRetriever(
        index=FaissIndex.build(np.eye(dim, dtype=np.float32)),
        documents=documents,
        embedder=_FixedEmbedder(),
    )
    return BM25Retriever(documents), dense


def _install_real_dense(monkeypatch, instance, *, loads=None):
    """Make ``retrievers.dense.DenseRetriever.load`` return a prebuilt real instance."""
    import retrievers.dense as dense_module

    class _PrebuiltDense:
        def __init__(self, *args, **kwargs):
            raise AssertionError("constructed directly instead of via load()")

        @classmethod
        def load(cls, directory, *, device=None):
            if loads is not None:
                loads.append((Path(directory), device))
            return instance

    monkeypatch.setattr(dense_module, "DenseRetriever", _PrebuiltDense)

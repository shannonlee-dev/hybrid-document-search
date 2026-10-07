"""A-type tests: build_search_service with doubles for BM25 and Dense.

These check initialization, availability states, Hybrid wiring, and lifecycle
rules without loading any real model or FAISS index. Doubles are installed with
monkeypatch, so the results do not depend on whether Role A or Role B has merged.
"""

import sys
import types
from pathlib import Path

import pytest

from app.schemas import RetrievalMethod
from app.service import MethodState, build_search_service
from data.loader import Document, load_prepared_documents
from retrievers.base import SearchResult

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "ko_miracl_prepared_corpus.jsonl"


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


class FakeTfidf:
    def __init__(self, documents):
        self.documents = tuple(documents)

    def search(self, query, top_k):
        return _ranked(self.documents, top_k, 1.0)


class FakeBM25:
    """Stands in for a merged BM25 retriever: has a constructor, so it is not a placeholder."""

    built_from = []

    def __init__(self, documents):
        self.documents = tuple(documents)
        FakeBM25.built_from.append(len(self.documents))

    def search(self, query, top_k):
        return _ranked(self.documents, top_k, 2.0)


def fake_dense_class(*, load_error=None, warm_error=None, documents_override=None):
    """Build a DenseRetriever double with the attributes the service reads."""

    class FakeDense:
        loads = []
        encode_queries = []

        def __init__(self, documents):
            self.documents = tuple(documents)
            self.embedder = self

        def encode_query(self, text):
            if warm_error is not None:
                raise warm_error
            FakeDense.encode_queries.append(text)
            return None

        def search(self, query, top_k):
            self.embedder.encode_query(query)
            return _ranked(self.documents, top_k, 3.0)

        @classmethod
        def load(cls, directory, *, device=None):
            cls.loads.append((Path(directory), device))
            if load_error is not None:
                raise load_error
            documents = documents_override or corpus_documents()
            return cls(documents)

    return FakeDense


def corpus_documents():
    return load_prepared_documents(FIXTURE_CORPUS)


@pytest.fixture
def fake_tfidf_module(monkeypatch):
    # Install the double as the module so this runs without scikit-learn.
    module = types.ModuleType("retrievers.tfidf")
    module.TfidfRetriever = FakeTfidf
    monkeypatch.setitem(sys.modules, "retrievers.tfidf", module)


@pytest.fixture
def fake_bm25(monkeypatch):
    FakeBM25.built_from.clear()
    import retrievers.bm25 as bm25_module

    monkeypatch.setattr(bm25_module, "BM25Retriever", FakeBM25)
    return FakeBM25


def _install_dense(monkeypatch, cls):
    import retrievers.dense as dense_module

    monkeypatch.setattr(dense_module, "DenseRetriever", cls)
    return cls


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


class _HideSklearn:
    """Make ``import sklearn`` fail the way a missing package does."""

    def find_spec(self, name, path=None, target=None):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ModuleNotFoundError(f"No module named {name!r}", name="sklearn")
        return None


def test_missing_sklearn_reports_dependency_state(monkeypatch):
    # Evict cached modules too, or an earlier import would satisfy the lookup.
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


def test_bm25_placeholder_reports_not_implemented(fake_tfidf_module):
    # The unmodified placeholder defines no constructor.
    service = build_search_service(FIXTURE_CORPUS)

    assert (
        service.availability()[RetrievalMethod.BM25].state
        is MethodState.NOT_IMPLEMENTED
    )


def test_dense_index_missing_is_reported_with_build_hint(
    monkeypatch, fake_tfidf_module, tmp_path
):
    _install_dense(monkeypatch, fake_dense_class(load_error=FileNotFoundError()))

    service = build_search_service(FIXTURE_CORPUS, dense_index_path=tmp_path / "idx")

    status = service.availability()[RetrievalMethod.DENSE]
    assert status.state is MethodState.INDEX_MISSING
    assert "scripts.build_index" in status.reason
    assert str(tmp_path) not in status.reason


def test_dense_dependency_missing_is_reported(monkeypatch, fake_tfidf_module, tmp_path):
    _install_dense(
        monkeypatch, fake_dense_class(load_error=ModuleNotFoundError("faiss"))
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
        monkeypatch, fake_dense_class(load_error=ValueError("checksum C:/secret"))
    )

    service = build_search_service(FIXTURE_CORPUS, dense_index_path=tmp_path)

    status = service.availability()[RetrievalMethod.DENSE]
    assert status.state is MethodState.LOAD_FAILED
    assert "C:/secret" not in status.reason


def test_dense_index_from_another_corpus_is_rejected(
    monkeypatch, fake_tfidf_module, tmp_path
):
    other = [Document("other-001#0", "다른 문서입니다.", None)]
    _install_dense(monkeypatch, fake_dense_class(documents_override=other))

    service = build_search_service(FIXTURE_CORPUS, dense_index_path=tmp_path)

    status = service.availability()[RetrievalMethod.DENSE]
    assert status.state is MethodState.LOAD_FAILED
    assert "different corpus" in status.reason


def test_model_load_failure_is_reported_at_startup_not_per_request(
    monkeypatch, fake_tfidf_module, tmp_path
):
    _install_dense(
        monkeypatch, fake_dense_class(warm_error=RuntimeError("model missing"))
    )

    service = build_search_service(FIXTURE_CORPUS, dense_index_path=tmp_path)

    status = service.availability()[RetrievalMethod.DENSE]
    assert status.state is MethodState.LOAD_FAILED
    assert "model missing" not in status.reason


def test_dense_loads_once_with_configured_device_and_warms_model(
    monkeypatch, fake_tfidf_module, tmp_path
):
    dense = _install_dense(monkeypatch, fake_dense_class())

    service = build_search_service(
        FIXTURE_CORPUS, dense_index_path=tmp_path, dense_device="cpu"
    )

    assert dense.loads == [(tmp_path, "cpu")]
    assert dense.encode_queries == ["warm-up"]
    assert service.availability()[RetrievalMethod.DENSE].available


def test_hybrid_is_bm25_plus_dense_with_rrf_when_both_available(
    monkeypatch, fake_tfidf_module, fake_bm25, tmp_path
):
    _install_dense(monkeypatch, fake_dense_class())

    service = build_search_service(FIXTURE_CORPUS, dense_index_path=tmp_path)

    assert service.availability()[RetrievalMethod.HYBRID].available
    hybrid = service.search("q", RetrievalMethod.HYBRID, 4)
    # Both doubles rank the corpus in the same order; RRF keeps that order and
    # preserves IDs and titles from the source results.
    expected_ids = [doc.document_id for doc in corpus_documents()]
    assert [hit.document_id for hit in hybrid] == expected_ids
    assert [hit.rank for hit in hybrid] == [1, 2, 3, 4]
    assert hybrid[0].title == "서울"


def test_hybrid_does_not_substitute_tfidf_for_missing_bm25(
    monkeypatch, fake_tfidf_module, tmp_path
):
    _install_dense(monkeypatch, fake_dense_class())

    service = build_search_service(FIXTURE_CORPUS, dense_index_path=tmp_path)

    status = service.availability()[RetrievalMethod.HYBRID]
    assert status.state is MethodState.UPSTREAM_UNAVAILABLE
    assert "BM25" in status.reason
    assert "TF-IDF" not in status.reason


def test_corpus_loaded_once_and_no_index_built_per_request(
    monkeypatch, fake_tfidf_module, fake_bm25, tmp_path
):
    import data.loader as loader

    calls = {"corpus": 0}
    real_load = loader.load_prepared_documents

    def counting_load(path):
        calls["corpus"] += 1
        return real_load(path)

    monkeypatch.setattr("app.service.load_prepared_documents", counting_load)
    dense = _install_dense(monkeypatch, fake_dense_class())
    service = build_search_service(FIXTURE_CORPUS, dense_index_path=tmp_path)
    builds_after_startup = len(fake_bm25.built_from)
    loads_after_startup = len(dense.loads)

    for _ in range(3):
        service.search("서울", RetrievalMethod.HYBRID, 2)
        service.search("서울", RetrievalMethod.DENSE, 2)

    assert calls["corpus"] == 1
    assert len(dense.loads) == loads_after_startup
    assert len(fake_bm25.built_from) == builds_after_startup
    # Startup embeds one warm-up string; each Dense or Hybrid call embeds only its
    # query (Hybrid calls Dense once per request). No corpus embedding happens here.
    assert dense.encode_queries[0] == "warm-up"
    assert len(dense.encode_queries) == 1 + 6

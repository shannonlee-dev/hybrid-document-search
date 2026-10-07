"""Integration tests: real TF-IDF over the tiny prepared fixture corpus.

These run offline, cross the real lifespan and service boundary, and need only the
sparse extra (scikit-learn). They do not download models or use the full dataset.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("sklearn")

import numpy as np  # noqa: E402

from app.api import create_app  # noqa: E402
from app.schemas import RetrievalMethod  # noqa: E402
from app.service import (  # noqa: E402
    MethodState,
    _is_placeholder,
    build_search_service,
)
from data.loader import load_prepared_documents  # noqa: E402

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "ko_miracl_prepared_corpus.jsonl"


def _client_with_corpus(monkeypatch, corpus: Path) -> TestClient:
    monkeypatch.setenv("SEARCH_CORPUS_PATH", str(corpus))
    return TestClient(create_app())


def test_tfidf_search_over_real_corpus_returns_stable_ids(monkeypatch):
    with _client_with_corpus(monkeypatch, FIXTURE_CORPUS) as client:
        response = client.post(
            "/search",
            json={"query": "대한민국의 수도", "method": "tfidf", "top_k": 3},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "tfidf"
    assert body["query"] == "대한민국의 수도"
    # The fixture's first passage is the only one about 서울 and 수도.
    assert body["results"][0]["document_id"] == "fixture-001#0"
    assert body["results"][0]["rank"] == 1
    assert body["results"][0]["title"] == "서울"
    assert all(hit["document_id"].startswith("fixture-") for hit in body["results"])
    assert len(body["results"]) <= 3


def test_tfidf_search_is_repeatable_across_requests(monkeypatch):
    with _client_with_corpus(monkeypatch, FIXTURE_CORPUS) as client:
        payload = {"query": "한글 문자", "method": "tfidf", "top_k": 4}
        first = client.post("/search", json=payload).json()
        second = client.post("/search", json=payload).json()

    assert first == second


def test_unavailable_methods_are_reported_over_real_startup(monkeypatch):
    with _client_with_corpus(monkeypatch, FIXTURE_CORPUS) as client:
        methods = {
            item["method"]: item
            for item in client.get("/search/methods").json()["methods"]
        }
        bm25 = client.post("/search", json={"query": "q", "method": "bm25"})

    assert methods["tfidf"]["available"] is True
    assert methods["bm25"]["available"] is False
    assert methods["hybrid"]["available"] is False
    assert bm25.status_code == 503


def test_missing_corpus_keeps_api_alive_and_reports_setup_hint(monkeypatch, tmp_path):
    missing = tmp_path / "absent.jsonl"

    with _client_with_corpus(monkeypatch, missing) as client:
        health = client.get("/health")
        search = client.post("/search", json={"query": "q", "method": "tfidf"})
        methods = client.get("/search/methods").json()["methods"]

    assert health.status_code == 200
    assert search.status_code == 503
    assert "prepare_dataset" in search.json()["detail"]
    assert str(missing) not in search.text
    assert all(item["available"] is False for item in methods)


def test_invalid_corpus_keeps_api_alive(monkeypatch, tmp_path):
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"document_id": "a", "text": ""}\n', encoding="utf-8")

    with _client_with_corpus(monkeypatch, broken) as client:
        health = client.get("/health")
        search = client.post("/search", json={"query": "q", "method": "tfidf"})

    assert health.status_code == 200
    assert search.status_code == 503
    assert "invalid" in search.json()["detail"]
    assert str(broken) not in search.text


# ---------------------------------------------------------------------------
# C-type real-implementation tests for Dense.
# These run only when the merged DenseRetriever exists and faiss is installed.
# The embedder is a deterministic local function, so no model is downloaded.
# ---------------------------------------------------------------------------

DENSE_DIM = 64


def _hash_embed(self, texts, prefix):
    """Character-bag embedding: deterministic, offline, and shares characters for overlap."""
    vectors = np.zeros((len(texts), DENSE_DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for char in text:
            vectors[row, ord(char) % DENSE_DIM] += 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


@pytest.fixture
def real_dense(monkeypatch, tmp_path):
    """Build and save a real Dense index with the local embedder; return its directory."""
    dense_module = pytest.importorskip("retrievers.dense")
    pytest.importorskip("faiss")
    if _is_placeholder(dense_module.DenseRetriever):
        pytest.skip("Dense implementation is not merged on this branch yet (Issue #3)")

    monkeypatch.setattr(dense_module.DenseEmbedder, "_encode", _hash_embed)
    documents = load_prepared_documents(FIXTURE_CORPUS)
    retriever = dense_module.DenseRetriever.build(
        documents, dense_module.DenseConfig(device="cpu")
    )
    index_dir = tmp_path / "dense-index"
    retriever.save(index_dir)
    return index_dir


def test_real_dense_loads_saved_index_without_rebuilding(monkeypatch, real_dense):
    from retrievers.dense import DenseRetriever

    # Loading and searching must never build or re-embed the corpus.
    monkeypatch.setattr(
        DenseRetriever,
        "build",
        classmethod(lambda cls, *a, **k: pytest.fail("rebuilt")),
    )

    service = build_search_service(
        FIXTURE_CORPUS, dense_index_path=real_dense, dense_device="cpu"
    )
    hits = service.search("서울", RetrievalMethod.DENSE, 3)

    assert service.availability()[RetrievalMethod.DENSE].available
    assert [hit.rank for hit in hits] == [1, 2, 3]
    # Results carry stable passage IDs, never FAISS row numbers.
    assert all("#" in hit.document_id for hit in hits)
    assert hits[0].document_id == "fixture-001#0"
    assert hits[0].title == "서울"


def test_real_dense_missing_index_reports_index_missing(tmp_path):
    dense_module = pytest.importorskip("retrievers.dense")
    if _is_placeholder(dense_module.DenseRetriever):
        pytest.skip("Dense implementation is not merged on this branch yet (Issue #3)")

    service = build_search_service(FIXTURE_CORPUS, dense_index_path=tmp_path / "none")

    status = service.availability()[RetrievalMethod.DENSE]
    assert status.state is MethodState.INDEX_MISSING


def test_real_dense_through_http_api(monkeypatch, real_dense):
    monkeypatch.setenv("SEARCH_CORPUS_PATH", str(FIXTURE_CORPUS))
    monkeypatch.setenv("DENSE_INDEX_PATH", str(real_dense))
    monkeypatch.setenv("DENSE_DEVICE", "cpu")

    with TestClient(create_app()) as client:
        response = client.post(
            "/search", json={"query": "서울", "method": "dense", "top_k": 2}
        )
        methods = {
            item["method"]: item
            for item in client.get("/search/methods").json()["methods"]
        }

    assert response.status_code == 200
    assert response.json()["results"][0]["document_id"] == "fixture-001#0"
    assert methods["dense"]["available"] is True
    # BM25 is still not merged, so Hybrid must stay unavailable.
    assert methods["hybrid"]["available"] is False
    assert methods["hybrid"]["state"] == "upstream_unavailable"

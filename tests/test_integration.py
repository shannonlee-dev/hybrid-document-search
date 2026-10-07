"""Integration tests: real TF-IDF over the tiny prepared fixture corpus.

These run offline, cross the real lifespan and service boundary, and need only the
sparse extra (scikit-learn). They do not download models or use the full dataset.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("sklearn")

from app.api import create_app  # noqa: E402

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

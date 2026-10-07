"""Offline FastAPI tests with an injected fake-backed SearchService."""

import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.schemas import RetrievalMethod
from app.service import MethodState, MethodStatus, SearchService
from retrievers.base import SearchResult

KOREAN_QUERY = "대한민국의 수도는 어디인가요?"


class FakeRetriever:
    def __init__(self, hits: list[tuple[str, str | None]]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return [
            SearchResult(
                document_id=doc_id,
                rank=rank,
                score=1.0 / rank,
                title=title,
                snippet=f"snippet {doc_id}",
            )
            for rank, (doc_id, title) in enumerate(self.hits[:top_k], start=1)
        ]


@pytest.fixture
def fake_tfidf() -> FakeRetriever:
    return FakeRetriever([("fixture-001#0", "서울"), ("fixture-002#0", None)])


@pytest.fixture
def client(fake_tfidf: FakeRetriever) -> TestClient:
    service = SearchService(
        {RetrievalMethod.TFIDF: fake_tfidf},
        statuses={
            RetrievalMethod.BM25: MethodStatus(
                MethodState.NOT_IMPLEMENTED, "BM25 retriever is not merged yet."
            ),
            RetrievalMethod.DENSE: MethodStatus(
                MethodState.NOT_IMPLEMENTED, "Dense retriever is not merged yet."
            ),
            RetrievalMethod.HYBRID: MethodStatus(
                MethodState.UPSTREAM_UNAVAILABLE,
                "requires BM25 and Dense, which is not available.",
            ),
        },
    )
    with TestClient(create_app(service=service)) as test_client:
        yield test_client


def test_health_is_unchanged(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_post_search_returns_schema_matching_response(client):
    response = client.post(
        "/search", json={"query": KOREAN_QUERY, "method": "tfidf", "top_k": 2}
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": KOREAN_QUERY,
        "method": "tfidf",
        "results": [
            {
                "document_id": "fixture-001#0",
                "rank": 1,
                "score": 1.0,
                "title": "서울",
                "snippet": "snippet fixture-001#0",
            },
            {
                "document_id": "fixture-002#0",
                "rank": 2,
                "score": 0.5,
                "title": None,
                "snippet": "snippet fixture-002#0",
            },
        ],
    }


def test_korean_query_is_forwarded_intact(client, fake_tfidf):
    client.post("/search", json={"query": KOREAN_QUERY, "method": "tfidf"})

    assert fake_tfidf.calls == [(KOREAN_QUERY, 10)]


def test_top_k_limits_results(client):
    response = client.post(
        "/search", json={"query": "q", "method": "tfidf", "top_k": 1}
    )

    assert [hit["document_id"] for hit in response.json()["results"]] == [
        "fixture-001#0"
    ]


def test_default_method_is_hybrid_which_is_unavailable(client):
    response = client.post("/search", json={"query": "q"})

    assert response.status_code == 503


@pytest.mark.parametrize(
    "body",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "q", "method": "colbert"},
        {"query": "q", "top_k": 0},
        {"query": "q", "top_k": -3},
        {"query": "q", "top_k": "5"},
        {"query": "q", "top_k": True},
        {"query": "q", "top_k": 1.5},
    ],
)
def test_invalid_requests_return_422(client, body):
    response = client.post("/search", json=body)

    assert response.status_code == 422


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid"])
def test_unconfigured_method_returns_503_not_empty_results(client, method):
    response = client.post("/search", json={"query": "q", "method": method})

    assert response.status_code == 503
    assert method in response.json()["detail"]
    assert "results" not in response.json()


def test_503_detail_does_not_include_filesystem_paths(client):
    response = client.post("/search", json={"query": "q", "method": "bm25"})

    assert "\\" not in response.text
    assert "/" not in response.json()["detail"]


def test_search_methods_lists_availability_and_reasons(client):
    response = client.get("/search/methods")

    assert response.status_code == 200
    methods = {item["method"]: item for item in response.json()["methods"]}
    assert set(methods) == {"tfidf", "bm25", "dense", "hybrid"}
    assert methods["tfidf"] == {
        "method": "tfidf",
        "available": True,
        "state": "available",
        "reason": None,
    }
    assert methods["bm25"]["available"] is False
    assert methods["bm25"]["state"] == "not_implemented"
    assert "not merged" in methods["bm25"]["reason"]


def test_uninitialized_service_returns_503_on_search():
    # Lifespan did not run, so no service was attached to app.state.
    app = create_app()
    client = TestClient(app)

    response = client.post("/search", json={"query": "q", "method": "tfidf"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Search service is not initialized."}


def test_importing_app_api_does_not_load_sparse_or_dense_stacks():
    # Run in a fresh interpreter so modules imported by other tests do not count.
    code = (
        "import sys, app.api; "
        "heavy = [m for m in ('sklearn', 'sentence_transformers', 'faiss') if m in sys.modules]; "
        "print(heavy)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"

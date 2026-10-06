"""Fast checks for the shared contract and service entry point."""

from fastapi.testclient import TestClient


def test_search_result_minimum_fields():
    from retrievers.base import SearchResult

    result = SearchResult(document_id="doc-1", rank=1, score=0.75)
    assert (result.document_id, result.rank, result.score) == ("doc-1", 1, 0.75)
    assert result.title is None
    assert result.snippet is None


def test_retriever_contract_accepts_independent_implementation():
    from retrievers.base import Retriever, SearchResult

    class ExampleRetriever:
        def search(self, query: str, top_k: int) -> list[SearchResult]:
            return [SearchResult(document_id=query, rank=1, score=1.0)][:top_k]

    retriever: Retriever = ExampleRetriever()
    assert retriever.search("doc-1", top_k=1)[0].document_id == "doc-1"


def test_health():
    from app.api import app

    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

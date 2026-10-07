"""Offline tests for the search request/response schemas."""

import pytest
from pydantic import ValidationError

from app.schemas import RetrievalMethod, SearchHit, SearchRequest, SearchResponse


def test_valid_request_with_explicit_values():
    req = SearchRequest(query="문서 검색", top_k=5, method="dense")

    assert req.query == "문서 검색"
    assert req.top_k == 5
    assert req.method is RetrievalMethod.DENSE


def test_request_defaults_to_top_k_10_and_hybrid():
    req = SearchRequest(query="q")

    assert req.top_k == 10
    assert req.method is RetrievalMethod.HYBRID


@pytest.mark.parametrize("method", ["tfidf", "bm25", "dense", "hybrid"])
def test_every_supported_method_is_accepted(method):
    assert SearchRequest(query="q", method=method).method.value == method


def test_unknown_method_is_rejected():
    with pytest.raises(ValidationError):
        SearchRequest(query="q", method="colbert")


@pytest.mark.parametrize("top_k", [0, -1])
def test_non_positive_top_k_is_rejected(top_k):
    with pytest.raises(ValidationError):
        SearchRequest(query="q", top_k=top_k)


def test_bool_top_k_is_rejected():
    with pytest.raises(ValidationError):
        SearchRequest(query="q", top_k=True)


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_is_rejected(query):
    with pytest.raises(ValidationError, match="blank"):
        SearchRequest(query=query)


def test_search_hit_metadata_is_optional():
    hit = SearchHit(document_id="doc-1", rank=1, score=0.5)

    assert (hit.title, hit.snippet) == (None, None)


def test_search_response_serializes_to_json_friendly_dict():
    response = SearchResponse(
        query="q",
        method=RetrievalMethod.HYBRID,
        results=[SearchHit(document_id="doc-1", rank=1, score=0.25, title="T")],
    )

    assert response.model_dump(mode="json") == {
        "query": "q",
        "method": "hybrid",
        "results": [
            {
                "document_id": "doc-1",
                "rank": 1,
                "score": 0.25,
                "title": "T",
                "snippet": None,
            }
        ],
    }

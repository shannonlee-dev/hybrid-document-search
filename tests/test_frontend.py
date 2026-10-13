"""Frontend tests over real HTTP: a local uvicorn server with fake retrievers."""

import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from streamlit.testing.v1 import AppTest

from app.api import create_app
from app.schemas import RetrievalMethod
from app.service import MethodState, MethodStatus, SearchService
from frontend.client import ApiError, fetch_methods, search
from retrievers.base import SearchResult

DENSE_REASON = "set DENSE_INDEX_PATH to a saved index directory."


class FakeRetriever:
    def __init__(self, doc_ids: list[str]) -> None:
        self.doc_ids = doc_ids
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return [
            SearchResult(
                document_id=doc_id,
                rank=rank,
                score=1.0 / rank,
                title=f"제목 {doc_id}",
                snippet=f"snippet {doc_id}",
            )
            for rank, doc_id in enumerate(self.doc_ids[:top_k], start=1)
        ]


def _serve(app: FastAPI):
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("test server did not start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, f"http://127.0.0.1:{port}"


@pytest.fixture
def retrievers():
    return {
        RetrievalMethod.TFIDF: FakeRetriever(["doc-1", "doc-2", "doc-3"]),
        RetrievalMethod.BM25: FakeRetriever(["doc-3", "doc-4"]),
    }


@pytest.fixture
def api_url(retrievers):
    service = SearchService(
        retrievers,
        statuses={
            RetrievalMethod.DENSE: MethodStatus(
                MethodState.NOT_CONFIGURED, DENSE_REASON
            ),
            RetrievalMethod.HYBRID: MethodStatus(
                MethodState.UPSTREAM_UNAVAILABLE,
                "requires Dense, which is not available.",
            ),
        },
    )
    server, thread, url = _serve(create_app(service=service))
    yield url
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def broken_api_url():
    """API that answers with unexpected bodies and a 500."""
    broken = FastAPI()
    broken.get("/search/methods")(lambda: {"unexpected": True})
    broken.post("/search")(lambda: PlainTextResponse("not json"))
    broken.get("/boom")(lambda: 1 / 0)
    server, thread, url = _serve(broken)
    yield url
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def closed_url():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def test_fetch_methods_reports_availability_and_reason(api_url):
    methods = {m.method: m for m in fetch_methods(api_url)}

    assert set(methods) == {"tfidf", "bm25", "dense", "hybrid"}
    assert methods["tfidf"].available and methods["bm25"].available
    assert not methods["dense"].available
    assert methods["dense"].state == "not_configured"
    assert methods["dense"].reason == DENSE_REASON


def test_search_sends_query_method_and_top_k(api_url, retrievers):
    hits = search(api_url, "대한민국의 수도", "tfidf", 2)

    assert retrievers[RetrievalMethod.TFIDF].calls == [("대한민국의 수도", 2)]
    assert retrievers[RetrievalMethod.BM25].calls == []
    assert [(h.rank, h.document_id, h.title) for h in hits] == [
        (1, "doc-1", "제목 doc-1"),
        (2, "doc-2", "제목 doc-2"),
    ]


def test_unavailable_method_shows_reason(api_url):
    with pytest.raises(ApiError, match="사용할 수 없습니다") as excinfo:
        search(api_url, "query", "dense", 5)
    assert DENSE_REASON in str(excinfo.value)


def test_invalid_request_shows_validation_message(api_url):
    with pytest.raises(ApiError, match="요청이 올바르지 않습니다") as excinfo:
        search(api_url, "   ", "tfidf", 5)
    assert "query must not be blank" in str(excinfo.value)


def test_server_down_shows_connection_hint(closed_url):
    with pytest.raises(ApiError, match="연결할 수 없습니다"):
        fetch_methods(closed_url)


def test_unexpected_responses_become_api_errors(broken_api_url):
    with pytest.raises(ApiError, match="응답 형식"):
        fetch_methods(broken_api_url)
    with pytest.raises(ApiError, match="응답 형식"):
        search(broken_api_url, "query", "tfidf", 5)


def test_server_error_hides_internal_details(broken_api_url):
    from frontend.client import _request

    with pytest.raises(ApiError) as excinfo:
        _request(f"{broken_api_url}/boom")
    assert str(excinfo.value) == "검색 API 오류가 발생했습니다 (HTTP 500)."


def _app(monkeypatch, url: str) -> AppTest:
    monkeypatch.setenv("SEARCH_API_URL", url)
    return AppTest.from_file("../frontend/app.py", default_timeout=30).run()


def test_ui_offers_only_available_methods(monkeypatch, api_url):
    at = _app(monkeypatch, api_url)

    assert not at.exception
    assert at.selectbox[0].options == ["TF-IDF", "BM25"]
    status = " ".join(md.value for md in at.markdown)
    assert "Dense" in status and "사용 불가" in status


def test_ui_single_search(monkeypatch, api_url, retrievers):
    at = _app(monkeypatch, api_url)
    at.text_input[0].input("서울")
    at.number_input[0].set_value(2)
    at.button[0].click().run()

    assert not at.exception
    assert retrievers[RetrievalMethod.TFIDF].calls == [("서울", 2)]
    text = " ".join(md.value for md in at.markdown)
    assert "제목 doc\\-1" in text and "제목 doc\\-3" not in text


def test_ui_compares_two_methods_by_rank(monkeypatch, api_url):
    at = _app(monkeypatch, api_url)
    at.toggle[0].set_value(True).run()
    at.text_input[0].input("서울")
    at.button[0].click().run()

    assert not at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "공통 문서 1개" in captions
    # doc-3 is TF-IDF rank 3 and BM25 rank 1.
    assert "상대 1위" in captions and "상대 3위" in captions


def test_ui_reports_server_down(monkeypatch, closed_url):
    at = _app(monkeypatch, closed_url)

    assert not at.exception
    assert "연결할 수 없습니다" in at.error[0].value


def test_non_http_api_url_is_rejected():
    with pytest.raises(ApiError, match="http://"):
        fetch_methods("file:///etc")


def test_overlong_query_is_rejected(api_url, retrievers):
    with pytest.raises(ApiError, match="요청이 올바르지 않습니다"):
        search(api_url, "가" * 1001, "tfidf", 5)
    assert retrievers[RetrievalMethod.TFIDF].calls == []


def test_ui_highlights_query_terms_and_logs_requests(monkeypatch, api_url):
    at = _app(monkeypatch, api_url)
    # "제목은" should still match "제목" in the fake titles.
    at.text_input[0].input("제목은 doc")
    at.button[0].click().run()

    assert not at.exception
    text = " ".join(md.value for md in at.markdown)
    assert ":orange-background[제목]" in text
    assert ":orange-background[doc]" in text
    details, log = (df.value for df in at.dataframe)
    assert list(details["일치한 검색어"])[0] == "제목, doc"
    assert list(log["상태"]) == ["성공"]

"""HTTP client for the search API used by the Streamlit UI.

The UI only talks to the public endpoints, so it never imports retrievers.
Every failure becomes ``ApiError`` with a message that is safe to show users.
"""

import json
import urllib.error
import urllib.request

from pydantic import BaseModel, ValidationError

TIMEOUT_SECONDS = 30


class ApiError(Exception):
    """Search API failure with a user-facing message."""


# Parsed locally instead of importing app.schemas: `streamlit run frontend/app.py`
# puts frontend/ first on sys.path, where `app` resolves to frontend/app.py.
class Hit(BaseModel):
    document_id: str
    rank: int
    score: float
    title: str | None = None
    snippet: str | None = None


class MethodInfo(BaseModel):
    method: str
    available: bool
    state: str
    reason: str | None = None


def fetch_methods(base_url: str) -> list[MethodInfo]:
    """Return every method's availability from ``GET /search/methods``."""
    data = _request(f"{base_url}/search/methods")
    try:
        return [MethodInfo.model_validate(item) for item in data["methods"]]
    except (KeyError, TypeError, ValidationError) as exc:
        raise ApiError("API 응답 형식이 올바르지 않습니다.") from exc


def search(base_url: str, query: str, method: str, top_k: int) -> list[Hit]:
    """Run ``POST /search`` and return the ranked hits."""
    payload = {"query": query, "method": method, "top_k": top_k}
    data = _request(f"{base_url}/search", payload)
    try:
        return [Hit.model_validate(item) for item in data["results"]]
    except (KeyError, TypeError, ValidationError) as exc:
        raise ApiError("API 응답 형식이 올바르지 않습니다.") from exc


def _request(url: str, payload: dict | None = None) -> object:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise ApiError(_http_error_message(exc)) from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise ApiError(
            "검색 API에 연결할 수 없습니다. 서버가 실행 중인지와 "
            "SEARCH_API_URL 설정을 확인하세요."
        ) from exc
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise ApiError("API 응답 형식이 올바르지 않습니다.") from exc


def _http_error_message(error: urllib.error.HTTPError) -> str:
    try:
        detail = json.loads(error.read()).get("detail")
    except (ValueError, AttributeError):
        detail = None
    if error.code == 422:
        # FastAPI validation detail: [{"loc": [...], "msg": "...", ...}, ...]
        messages = [
            item["msg"]
            for item in detail or []
            if isinstance(item, dict) and "msg" in item
        ]
        return "요청이 올바르지 않습니다." + (
            f" ({'; '.join(messages)})" if messages else ""
        )
    if error.code == 503 and isinstance(detail, str):
        # The service already keeps paths and stack traces out of 503 reasons.
        return f"검색 방식을 사용할 수 없습니다: {detail}"
    return f"검색 API 오류가 발생했습니다 (HTTP {error.code})."

"""NAVER Cloud RAG adapter (VPC, Korea region) using only the documented REST API.

Checked against https://api.ncloud-docs.com/docs/rag-overview on 2026-10-29:

- documents are added with a caller-chosen ID (`POST /doc/{docId}`), so the
  BC Card `source_chunk_id` is the provider document ID and is repeated in
  `extra.source_chunk_id`;
- there is no retrieval-only endpoint. `POST /conversation` runs the whole
  pipeline (QUERY_REWRITE, RETRIEVAL, GENERATOR); `verbose: true` exposes the
  RETRIEVAL block, whose chunks carry `docId`, `score`, `rank` and `chunk_no`.
"""

import json
import os
import socket
import urllib.error
import urllib.request
from urllib.parse import quote

from commercial.base import ProviderError, ProviderHit, ProviderResponse

DEFAULT_BASE_URL = "https://kr-pub-gateway.rag.naverncp.com"
API_KEY_ENV = "NAVER_RAG_API_KEY"
SERVICE_ID_ENV = "NAVER_RAG_SERVICE_ID"
BASE_URL_ENV = "NAVER_RAG_BASE_URL"
# 문서상 최대 chunkSize와 overlap 0으로 BC Card chunk(최대 2,307자)를 가능한 한 쪼개지 않는다.
# test 결과를 보기 전에 문서만 보고 정한 값이며 이후 바꾸지 않는다.
RETRIEVAL_PARAMETERS = {"chunkSize": 4096, "overlapSize": 0}
DEFAULT_TIMEOUT_SECONDS = 60.0
_RATE_LIMIT_CODES = {50003}
_QUOTA_CODES = {50009}


def urllib_transport(method, url, headers, body, timeout):
    """Return (status, headers, body bytes); HTTP errors are returned, not raised."""
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()
    except (TimeoutError, socket.timeout) as exc:
        raise ProviderError(str(exc), kind="timeout", transient=True) from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise ProviderError(str(exc), kind="timeout", transient=True) from exc
        raise ProviderError(str(exc), kind="network", transient=True) from exc


class NaverRagClient:
    name = "naver_cloud_rag"

    def __init__(
        self,
        api_key,
        service_id,
        *,
        base_url=DEFAULT_BASE_URL,
        transport=urllib_transport,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    ):
        if not api_key or not service_id:
            raise ValueError(f"{API_KEY_ENV}와 {SERVICE_ID_ENV}가 필요합니다.")
        self._api_key = api_key
        self.service_id = service_id
        self.base_url = base_url.rstrip("/")
        self._transport = transport
        self.timeout = timeout

    @classmethod
    def from_env(cls, environ=os.environ, **kwargs):
        return cls(
            environ.get(API_KEY_ENV),
            environ.get(SERVICE_ID_ENV),
            base_url=environ.get(BASE_URL_ENV, DEFAULT_BASE_URL),
            **kwargs,
        )

    def describe(self):
        return {
            "provider": "NAVER Cloud Platform",
            "product": "RAG (VPC)",
            "api": "RAG API v1 (/api/v1/svc/{serviceId})",
            "region": "KR",
            "base_url": self.base_url,
            "service_id": self.service_id,
            "endpoint": "POST /conversation, verbose=true, RETRIEVAL block",
            "retrieval_parameters": RETRIEVAL_PARAMETERS,
            "generator": "service default (not retrieval-only; always runs)",
            "query_rewrite": "service pipeline; not controllable through the API",
        }

    def _request(self, method, path, payload=None):
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/api/v1/svc/{quote(self.service_id, safe='')}{path}"
        status, response_headers, raw = self._transport(
            method, url, headers, body, self.timeout
        )
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            data = None
        _raise_for_error(status, response_headers, data)
        if not isinstance(data, dict):
            raise ProviderError("JSON 객체가 아닌 응답", kind="malformed_response")
        return data

    def add_document(self, document_id, text):
        """Index one BC Card chunk under its own ID; the title is not indexed."""
        payload = {"body": text, "extra": {"source_chunk_id": document_id}}
        data = self._request("POST", f"/doc/{quote(document_id, safe='')}", payload)
        inserted = (data.get("result") or {}).get("inserted") or []
        if document_id not in inserted:
            raise ProviderError(
                f"요청한 ID가 inserted에 없습니다: {inserted!r}",
                kind="id_not_preserved",
                abort=True,
            )
        return data

    def document_status(self, document_id):
        data = self._request("GET", f"/doc/{quote(document_id, safe='')}/status")
        return (data.get("data") or {}).get("status")

    def delete_document(self, document_id):
        return self._request("DELETE", f"/doc/{quote(document_id, safe='')}")

    def document_count(self):
        return (self._request("GET", "/doc-count").get("result") or {}).get(
            "total_count"
        )

    def search(self, query, top_k):
        payload = {
            "messages": [{"role": "user", "content": query}],
            "blocks": {
                "RETRIEVAL": {
                    "type": "RETRIEVAL",
                    "parameters": {"k": top_k, **RETRIEVAL_PARAMETERS},
                }
            },
            "verbose": True,
        }
        data = self._request("POST", "/conversation", payload)
        return parse_conversation(data)


def _raise_for_error(status, headers, data):
    code = data.get("code") if isinstance(data, dict) else None
    status_block = data.get("status") if isinstance(data, dict) else None
    if code is None and isinstance(status_block, dict):
        code = status_block.get("code")
    message = data.get("message") if isinstance(data, dict) else None
    if 200 <= status < 300 and (code is None or code in (0, 200)):
        return
    detail = f"HTTP {status}, code {code}: {message}"
    retry_after = _retry_after(headers)
    if status == 429 or code in _RATE_LIMIT_CODES:
        raise ProviderError(
            detail,
            kind="rate_limited",
            transient=True,
            retry_after=retry_after,
            status=status,
        )
    if code in _QUOTA_CODES:
        raise ProviderError(detail, kind="quota_exceeded", abort=True, status=status)
    if status in (401, 403):
        raise ProviderError(detail, kind="auth", abort=True, status=status)
    if status == 404:
        raise ProviderError(detail, kind="not_found", abort=True, status=status)
    if status in (408, 504) or status >= 500:
        raise ProviderError(detail, kind="server_error", transient=True, status=status)
    raise ProviderError(detail, kind="request_rejected", status=status)


def _retry_after(headers):
    for key, value in (headers or {}).items():
        if key.lower() == "retry-after":
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                return None
    return None


def _field(item, *names):
    for name in names:
        if item.get(name) is not None:
            return item[name]
    return None


def parse_conversation(data):
    """Extract RETRIEVAL-block chunks; snake_case and camelCase both appear in docs."""
    results = (data.get("data") or {}).get("results")
    if not isinstance(results, list):
        raise ProviderError("data.results가 없습니다.", kind="malformed_response")
    blocks = [
        block
        for block in results
        if isinstance(block, dict) and block.get("blockId") == "RETRIEVAL"
    ]
    if len(blocks) != 1:
        raise ProviderError(
            "RETRIEVAL block이 정확히 하나여야 합니다(verbose=true).",
            kind="malformed_response",
        )
    value = blocks[0].get("value")
    if value is None or value == "":
        value = []
    if not isinstance(value, list):
        raise ProviderError(
            "RETRIEVAL value가 목록이 아닙니다.", kind="malformed_response"
        )
    hits = []
    for item in value:
        if not isinstance(item, dict):
            raise ProviderError(
                "RETRIEVAL 항목이 객체가 아닙니다.", kind="malformed_response"
            )
        extra = item.get("extra") if isinstance(item.get("extra"), dict) else {}
        score = _field(item, "score")
        rank = _field(item, "rank")
        chunk = _field(item, "chunk_no", "chunkNo")
        hits.append(
            ProviderHit(
                provider_id=_field(item, "docId", "doc_id", "docid"),
                rank=rank
                if isinstance(rank, int) and not isinstance(rank, bool)
                else None,
                score=float(score) if isinstance(score, (int, float)) else None,
                original_id=extra.get("source_chunk_id"),
                chunk=None if chunk is None else str(chunk),
            )
        )
    usage = {}
    for block in results:
        if isinstance(block, dict) and block.get("blockId") == "GENERATOR":
            metadata = block.get("metadata") or {}
            usage = {
                key: metadata.get(key)
                for key in ("modelName", "inputTokens", "outputTokens", "totalTokens")
                if key in metadata
            }
    return ProviderResponse(hits=tuple(hits), raw=data, usage=usage)

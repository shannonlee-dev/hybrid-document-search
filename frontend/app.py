"""Streamlit search UI (Role 3). Searches only through the FastAPI endpoints.

Run from the repository root:
    uv run streamlit run frontend/app.py
Environment:
    SEARCH_API_URL  search API base URL (default http://127.0.0.1:8000)
"""

import os
import re
import time

import streamlit as st

from frontend.client import ApiError, fetch_methods, search

DEFAULT_API_URL = "http://127.0.0.1:8000"
MAX_QUERY_LENGTH = 1000  # app.schemas.MAX_QUERY_LENGTH; app.* is not importable here
LABELS = {"tfidf": "TF-IDF", "bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid"}


def _escape(text: str) -> str:
    """Show titles and snippets as plain text inside st.markdown."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$:])", r"\\\1", text)


def _label(method: str) -> str:
    return LABELS.get(method, method)


def _matched_terms(query: str, text: str) -> list[str]:
    """Query words found in ``text``.

    Shorter prefixes (down to 2 chars) are tried so Korean particles still match:
    "수도는" -> "수도". A one-character word like "섬" must match as is.
    """
    lowered = text.lower()
    terms = []
    for word in re.findall(r"\w+", query.lower()):
        for end in range(len(word), min(len(word), 2) - 1, -1):
            if word[:end] in lowered:
                terms.append(word[:end])
                break
    return terms


def _highlight(text: str, terms: list[str]) -> str:
    if not terms:
        return _escape(text)
    pattern = re.compile(
        "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True)),
        re.IGNORECASE,
    )
    parts, last = [], 0
    for match in pattern.finditer(text):
        parts.append(_escape(text[last : match.start()]))
        parts.append(f":orange-background[{_escape(match.group())}]")
        last = match.end()
    parts.append(_escape(text[last:]))
    return "".join(parts)


def _render_hits(query: str, hits, other_ranks: dict[str, int] | None = None) -> None:
    if not hits:
        st.info("검색 결과가 없습니다.")
        return
    for hit in hits:
        terms = _matched_terms(query, f"{hit.title or ''} {hit.snippet or ''}")
        title = _highlight(hit.title, terms) if hit.title else _escape(hit.document_id)
        st.markdown(f"**{hit.rank}. {title}**")
        meta = f"`{hit.document_id}` · score {hit.score:.4f}"
        if other_ranks is not None:
            other = other_ranks.get(hit.document_id)
            meta += f" · 상대 {other}위" if other else " · 상대 결과에 없음"
        st.caption(meta)
        if hit.snippet:
            st.markdown(_highlight(hit.snippet, terms))


def _run_search(api_url: str, query: str, method: str, top_k: int):
    """Search and record the request in the session log shown at the bottom."""
    started = time.perf_counter()
    try:
        hits = search(api_url, query, method, top_k)
        status = "성공"
    except ApiError as exc:
        st.error(str(exc))
        hits, status = None, str(exc)
    st.session_state.log.append(
        {
            "시각": time.strftime("%H:%M:%S"),
            "방식": _label(method),
            "검색어": query if len(query) <= 40 else query[:40] + "…",
            "top_k": top_k,
            "결과 수": len(hits) if hits is not None else None,
            "응답 ms": round((time.perf_counter() - started) * 1000, 1),
            "상태": status,
        }
    )
    for hit in hits or []:
        terms = _matched_terms(query, f"{hit.title or ''} {hit.snippet or ''}")
        st.session_state.details.append(
            {
                "방식": _label(method),
                "rank": hit.rank,
                "document_id": hit.document_id,
                "title": hit.title,
                "일치한 검색어": ", ".join(dict.fromkeys(terms)) or "-",
            }
        )
    return hits


def _render_log() -> None:
    st.divider()
    st.subheader("검색 로그")
    st.caption(
        "응답 ms는 UI에서 잰 API 왕복 시간입니다. "
        "서버 쪽 검색 시간은 API 서버 로그(`search method=...`)에 남습니다."
    )
    if st.session_state.details:
        st.markdown("**최근 검색 세부 정보**")
        st.caption(
            "일치한 검색어는 제목·snippet에 그대로 나온 단어입니다. "
            "Dense는 단어가 겹치지 않아도 의미로 찾을 수 있습니다."
        )
        st.dataframe(st.session_state.details, hide_index=True)
    if st.session_state.log:
        st.markdown("**이번 세션 요청**")
        st.dataframe(st.session_state.log[::-1], hide_index=True)
    else:
        st.caption("아직 검색 기록이 없습니다.")


st.set_page_config(page_title="Hybrid Document Search", layout="wide")
st.title("Hybrid Document Search")
st.session_state.setdefault("log", [])

api_url = (os.environ.get("SEARCH_API_URL") or DEFAULT_API_URL).rstrip("/")

try:
    methods = fetch_methods(api_url)
except ApiError as exc:
    st.error(str(exc))
    st.caption(f"SEARCH_API_URL: {api_url}")
    st.stop()

available = [m.method for m in methods if m.available]
with st.expander("검색 방식 상태", expanded=not available):
    for m in methods:
        status = "사용 가능" if m.available else f"사용 불가 — {m.reason or m.state}"
        st.markdown(f"- **{_label(m.method)}**: {_escape(status)}")

if not available:
    st.warning("사용 가능한 검색 방식이 없습니다. API 서버 설정을 확인하세요.")
    st.stop()

compare = st.toggle("두 방식 비교", disabled=len(available) < 2)

with st.form("search"):
    query = st.text_input("검색어", max_chars=MAX_QUERY_LENGTH)
    if compare:
        left, right = st.columns(2)
        method_a = left.selectbox("방식 A", available, format_func=_label)
        method_b = right.selectbox("방식 B", available, index=1, format_func=_label)
    else:
        method_a = st.selectbox("검색 방식", available, format_func=_label)
    top_k = st.number_input("top_k", min_value=1, max_value=100, value=10)
    submitted = st.form_submit_button("검색")

if submitted:
    st.session_state.details = []
    if not query.strip():
        st.warning("검색어를 입력하세요.")
    elif not compare:
        hits = _run_search(api_url, query, method_a, int(top_k))
        if hits is not None:
            _render_hits(query, hits)
    elif method_a == method_b:
        st.warning("서로 다른 두 방식을 선택하세요.")
    else:
        summary = st.empty()
        columns = st.columns(2)
        results = {}
        for column, method in zip(columns, (method_a, method_b)):
            with column:
                st.subheader(_label(method))
                results[method] = _run_search(api_url, query, method, int(top_k))
        ranks = {
            method: {hit.document_id: hit.rank for hit in hits}
            for method, hits in results.items()
            if hits is not None
        }
        for column, method, other in zip(
            columns, (method_a, method_b), (method_b, method_a)
        ):
            if results[method] is not None:
                with column:
                    _render_hits(query, results[method], ranks.get(other))
        if len(ranks) == 2:
            common = ranks[method_a].keys() & ranks[method_b].keys()
            summary.caption(
                f"공통 문서 {len(common)}개. "
                "score는 방식마다 의미와 범위가 달라 서로 비교하지 않습니다."
            )

st.session_state.setdefault("details", [])
_render_log()

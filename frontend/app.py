"""Streamlit search UI (Role 3). Searches only through the FastAPI endpoints.

Run from the repository root:
    uv run streamlit run frontend/app.py
Environment:
    SEARCH_API_URL  search API base URL (default http://127.0.0.1:8000)
"""

import os
import re

import streamlit as st

from frontend.client import ApiError, fetch_methods, search

DEFAULT_API_URL = "http://127.0.0.1:8000"
LABELS = {"tfidf": "TF-IDF", "bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid"}


def _escape(text: str) -> str:
    """Show titles and snippets as plain text inside st.markdown."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$])", r"\\\1", text)


def _label(method: str) -> str:
    return LABELS.get(method, method)


def _render_hits(hits, other_ranks: dict[str, int] | None = None) -> None:
    if not hits:
        st.info("검색 결과가 없습니다.")
        return
    for hit in hits:
        st.markdown(f"**{hit.rank}. {_escape(hit.title or hit.document_id)}**")
        meta = f"`{hit.document_id}` · score {hit.score:.4f}"
        if other_ranks is not None:
            other = other_ranks.get(hit.document_id)
            meta += f" · 상대 {other}위" if other else " · 상대 결과에 없음"
        st.caption(meta)
        if hit.snippet:
            st.markdown(_escape(hit.snippet))


def _run_search(api_url: str, query: str, method: str, top_k: int):
    try:
        return search(api_url, query, method, top_k)
    except ApiError as exc:
        st.error(str(exc))
        return None


st.set_page_config(page_title="Hybrid Document Search", layout="wide")
st.title("Hybrid Document Search")

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
    query = st.text_input("검색어")
    if compare:
        left, right = st.columns(2)
        method_a = left.selectbox("방식 A", available, format_func=_label)
        method_b = right.selectbox("방식 B", available, index=1, format_func=_label)
    else:
        method_a = st.selectbox("검색 방식", available, format_func=_label)
    top_k = st.number_input("top_k", min_value=1, max_value=100, value=10)
    submitted = st.form_submit_button("검색")

if submitted:
    if not query.strip():
        st.warning("검색어를 입력하세요.")
    elif not compare:
        hits = _run_search(api_url, query, method_a, int(top_k))
        if hits is not None:
            _render_hits(hits)
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
                    _render_hits(results[method], ranks.get(other))
        if len(ranks) == 2:
            common = ranks[method_a].keys() & ranks[method_b].keys()
            summary.caption(
                f"공통 문서 {len(common)}개. "
                "score는 방식마다 의미와 범위가 달라 서로 비교하지 않습니다."
            )

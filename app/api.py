"""FastAPI entry point (Role 3).

Importing this module loads no corpus, index, or model. The search service is
built in the lifespan handler when the server starts. Tests pass a preconfigured
service to ``create_app(service=...)``.

Run from the repository root:
    uv run --extra sparse uvicorn app.api:app
Environment:
    SEARCH_CORPUS_PATH  corpus JSONL (default data/processed/corpus.jsonl)
    DENSE_INDEX_PATH    saved Dense index directory (Dense is off when unset)
    DENSE_DEVICE        device for the Dense model, e.g. cpu or cuda (optional)
"""

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request

from app.schemas import SearchHit, SearchRequest, SearchResponse
from app.service import MethodUnavailableError, SearchService, build_search_service

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_PATH = Path("data/processed/corpus.jsonl")


def create_app(service: SearchService | None = None) -> FastAPI:
    """Build the app. Pass ``service`` to skip corpus loading (used by tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is None:
            app.state.search_service = build_search_service(
                Path(os.environ.get("SEARCH_CORPUS_PATH", DEFAULT_CORPUS_PATH)),
                dense_index_path=_env_path("DENSE_INDEX_PATH"),
                dense_device=os.environ.get("DENSE_DEVICE") or None,
            )
        yield

    app = FastAPI(title="Hybrid Document Search", lifespan=lifespan)
    if service is not None:
        app.state.search_service = service

    def get_search_service(request: Request) -> SearchService:
        service_state = getattr(request.app.state, "search_service", None)
        if service_state is None:
            raise HTTPException(
                status_code=503, detail="Search service is not initialized."
            )
        return service_state

    @app.get("/health")
    def health() -> dict[str, str]:
        """Confirm that the API process is running. Does not check search readiness."""
        return {"status": "ok"}

    @app.get("/search/methods")
    def search_methods(
        search: SearchService = Depends(get_search_service),
    ) -> dict[str, list[dict[str, str | bool | None]]]:
        """List each retrieval method's state and, when unavailable, the reason."""
        return {
            "methods": [
                {
                    "method": method.value,
                    "available": status.available,
                    "state": status.state.value,
                    "reason": status.reason,
                }
                for method, status in search.availability().items()
            ]
        }

    @app.post("/search", response_model=SearchResponse)
    def search(
        body: SearchRequest,
        search: SearchService = Depends(get_search_service),
    ) -> SearchResponse:
        """Run one retrieval method. Returns 503 if that method is unavailable."""
        started = time.perf_counter()
        try:
            results = search.search(body.query, body.method, body.top_k)
        except MethodUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        # Query text is not logged. Latency here is query-time only, not index build.
        logger.info(
            "search method=%s top_k=%d hits=%d latency_ms=%.2f",
            body.method.value,
            body.top_k,
            len(results),
            latency_ms,
        )
        return SearchResponse(
            query=body.query,
            method=body.method,
            results=[
                SearchHit(
                    document_id=result.document_id,
                    rank=result.rank,
                    score=result.score,
                    title=result.title,
                    snippet=result.snippet,
                )
                for result in results
            ],
        )

    return app


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


app = create_app()

"""FastAPI search contracts; SearchHit mirrors SearchResult fields."""

from enum import StrEnum

from pydantic import BaseModel, Field, StrictInt, field_validator


class RetrievalMethod(StrEnum):
    TFIDF = "tfidf"
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


MAX_QUERY_LENGTH = 1000


class SearchRequest(BaseModel):
    # Long queries tie up a worker: a 3.8MB query took 76s on Dense and 100s on Hybrid.
    query: str = Field(max_length=MAX_QUERY_LENGTH)
    top_k: StrictInt = Field(default=10, gt=0)
    method: RetrievalMethod = RetrievalMethod.HYBRID

    @field_validator("query")
    @classmethod
    def _reject_blank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class SearchHit(BaseModel):
    document_id: str
    rank: StrictInt = Field(gt=0)
    score: float
    title: str | None = None
    snippet: str | None = None


class SearchResponse(BaseModel):
    query: str
    method: RetrievalMethod
    results: list[SearchHit]

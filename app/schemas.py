"""FastAPI search contracts; SearchHit mirrors SearchResult fields."""

from enum import StrEnum

from pydantic import BaseModel, Field, StrictInt, field_validator


class RetrievalMethod(StrEnum):
    TFIDF = "tfidf"
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


class SearchRequest(BaseModel):
    query: str
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

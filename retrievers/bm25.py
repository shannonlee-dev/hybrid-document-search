"""Retrieve Korean text with BM25 using the shared character n-gram analyzer."""

from collections.abc import Iterable
from heapq import nsmallest
from math import isfinite
from numbers import Real

import bm25s
from sklearn.feature_extraction.text import TfidfVectorizer

from data.loader import Document
from data.preprocess import build_index_text
from retrievers.base import SNIPPET_LENGTH, SearchResult

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
SCORING_METHOD = "lucene"


class BM25Retriever:
    """Index shared documents and rank hits by raw BM25 score."""

    def __init__(
        self,
        documents: Iterable[Document],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        if (
            isinstance(k1, bool)
            or not isinstance(k1, Real)
            or not isfinite(k1)
            or k1 <= 0
        ):
            raise ValueError("k1은 유한한 양수여야 합니다.")
        if (
            isinstance(b, bool)
            or not isinstance(b, Real)
            or not isfinite(b)
            or not 0 <= b <= 1
        ):
            raise ValueError("b는 0 이상 1 이하의 유한한 수여야 합니다.")

        self._documents = tuple(documents)
        if not self._documents:
            raise ValueError("인덱싱할 문서가 하나 이상 필요합니다.")
        document_ids = [document.document_id for document in self._documents]
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("중복 document_id가 있습니다.")

        texts = [build_index_text(document) for document in self._documents]
        if any(not text.strip() for text in texts):
            raise ValueError("인덱싱할 텍스트는 비어 있으면 안 됩니다.")

        # TF-IDF와 동일한 분석기로 토큰만 생성하며 TF-IDF 가중치는 사용하지 않습니다.
        self._analyze = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4)
        ).build_analyzer()
        tokens = [self._analyze(text) for text in texts]
        self._index = bm25s.BM25(
            k1=float(k1), b=float(b), method=SCORING_METHOD, idf_method=SCORING_METHOD
        )
        self._index.index(tokens, show_progress=False)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return at most top_k hits with positive BM25 scores."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k는 양의 정수여야 합니다.")
        if not isinstance(query, str):
            raise TypeError("query는 문자열이어야 합니다.")
        if not query.strip():
            return []

        tokens = self._analyze(query)
        if not tokens:
            return []
        scores = self._index.get_scores(tokens)
        candidates = (
            (index, float(score)) for index, score in enumerate(scores) if score > 0
        )
        # TF-IDF와 동일하게 동점은 입력 corpus 순서로 처리합니다.
        selected = nsmallest(top_k, candidates, key=lambda item: (-item[1], item[0]))
        results = []
        for rank, (index, score) in enumerate(selected, start=1):
            document = self._documents[index]
            results.append(
                SearchResult(
                    document_id=document.document_id,
                    rank=rank,
                    score=score,
                    title=document.title,
                    snippet=document.text[:SNIPPET_LENGTH],
                )
            )
        return results

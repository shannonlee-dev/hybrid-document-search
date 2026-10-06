"""한국어 문자 n-gram 기반 TF-IDF 검색을 제공합니다."""

from collections.abc import Iterable
from heapq import nsmallest

from sklearn.feature_extraction.text import TfidfVectorizer

from data.loader import Document
from data.preprocess import build_index_text
from retrievers.base import SearchResult


class TfidfRetriever:
    """문서로 인덱스를 만들고 공통 검색 계약에 맞춰 결과를 반환합니다."""

    def __init__(self, documents: Iterable[Document]) -> None:
        self._documents = tuple(documents)
        if not self._documents:
            raise ValueError("인덱싱할 문서가 하나 이상 필요합니다.")

        document_ids = [document.document_id for document in self._documents]
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("중복 document_id가 있습니다.")

        texts = [build_index_text(document) for document in self._documents]
        if any(not text.strip() for text in texts):
            raise ValueError("인덱싱할 텍스트는 비어 있으면 안 됩니다.")

        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            norm="l2",
        )
        self._matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """양의 점수를 가진 결과를 최대 top_k개까지 좋은 순서로 반환합니다."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k는 양의 정수여야 합니다.")
        if not isinstance(query, str):
            raise TypeError("query는 문자열이어야 합니다.")
        if not query.strip():
            return []

        query_vector = self._vectorizer.transform([query])
        if query_vector.nnz == 0:
            return []

        # L2 정규화된 벡터의 내적은 코사인 유사도와 같습니다.
        scores = (self._matrix @ query_vector.T).tocoo()
        candidates = (
            (int(row), float(score))
            for row, score in zip(scores.row, scores.data, strict=True)
            if score > 0
        )
        # 점수가 같으면 원본 corpus 순서를 사용합니다.
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
                    snippet=document.text[:200],
                )
            )
        return results

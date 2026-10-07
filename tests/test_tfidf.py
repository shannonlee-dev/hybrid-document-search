"""TF-IDF 검색의 순위, 입력 검증과 공통 결과 계약을 검증합니다."""

from pathlib import Path

import pytest

from data.loader import Document, load_documents
from retrievers.base import Retriever, SearchResult

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ko_miracl_corpus.jsonl"


@pytest.fixture
def retriever_class():
    # 기본 CI에는 Sparse extra가 없으므로 검색 테스트만 건너뜁니다.
    pytest.importorskip("sklearn")

    from retrievers.tfidf import TfidfRetriever

    return TfidfRetriever


@pytest.fixture
def retriever(retriever_class):
    return retriever_class(load_documents(FIXTURE_PATH))


@pytest.mark.parametrize(
    ("query", "expected_id"),
    [
        ("수도", "fixture-001#0"),
        ("제주", "fixture-002#0"),
        ("한글", "fixture-003#0"),
        ("부산", "fixture-004#0"),
    ],
)
def test_korean_top_hit(retriever, query, expected_id):
    results = retriever.search(query, top_k=1)

    assert len(results) == 1
    assert results[0].document_id == expected_id
    assert results[0].rank == 1
    assert results[0].score > 0


def test_top_k_ranking_and_scores(retriever):
    results = retriever.search("대한민국", top_k=2)

    assert len(results) == 2
    assert [result.rank for result in results] == [1, 2]
    assert results[0].score >= results[1].score > 0
    assert len({result.document_id for result in results}) == 2


def test_top_k_can_exceed_corpus_size(retriever):
    results = retriever.search("대한민국", top_k=100)

    assert 0 < len(results) <= 4
    assert [result.rank for result in results] == list(range(1, len(results) + 1))
    assert all(result.score > 0 for result in results)


@pytest.mark.parametrize("query", ["", "   ", "\n\t", "zzzzzz"])
def test_empty_or_out_of_vocabulary_query(retriever, query):
    assert retriever.search(query, top_k=3) == []


@pytest.mark.parametrize("top_k", [0, -1, 1.5, "2", None, True])
def test_invalid_top_k(retriever, top_k):
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("서울", top_k=top_k)


@pytest.mark.parametrize("query", [None, 123])
def test_invalid_query_type(retriever, query):
    with pytest.raises(TypeError, match="query"):
        retriever.search(query, top_k=1)


def test_empty_corpus(retriever_class):
    with pytest.raises(ValueError, match="문서"):
        retriever_class([])


def test_duplicate_document_ids(retriever_class):
    documents = [Document("same#0", "서울"), Document("same#0", "부산")]

    with pytest.raises(ValueError, match="중복 document_id"):
        retriever_class(documents)


def test_empty_index_text(retriever_class):
    with pytest.raises(ValueError, match="텍스트"):
        retriever_class([Document("doc#0", "   ")])


def test_ties_use_corpus_order(retriever_class):
    documents = [
        Document("z#0", "동일한 본문"),
        Document("a#0", "동일한 본문"),
    ]
    retriever = retriever_class(documents)

    first = retriever.search("동일한", top_k=2)
    second = retriever.search("동일한", top_k=2)

    assert [result.document_id for result in first] == ["z#0", "a#0"]
    assert first == second
    assert first[0].score == pytest.approx(first[1].score)


def test_title_is_searchable(retriever_class):
    documents = [
        Document("doc#0", "관련 본문입니다.", "제주"),
        Document("doc#1", "다른 본문입니다.", None),
    ]

    results = retriever_class(documents).search("제주", top_k=1)

    assert results[0].document_id == "doc#0"
    assert results[0].title == "제주"
    assert results[0].snippet == "관련 본문입니다."


def test_search_preserves_original_text(retriever_class):
    text = "  원본 공백\n둘째 줄  "
    document = Document("doc#12", text, "서울")

    results = retriever_class([document]).search("서울", top_k=1)

    assert results[0].snippet == text
    assert document.text == text
    assert results[0].document_id == "doc#12"


def test_zero_score_documents_are_excluded(retriever_class):
    documents = [Document("ko#0", "서울"), Document("en#0", "zzzzzz")]

    results = retriever_class(documents).search("서울", top_k=10)

    assert [result.document_id for result in results] == ["ko#0"]


def test_shared_retriever_contract(retriever):
    shared: Retriever = retriever
    result = shared.search("서울", top_k=1)[0]

    assert isinstance(result, SearchResult)
    assert result.document_id == "fixture-001#0"
    assert result.title == "서울"
    assert result.snippet == "서울은 대한민국의 수도입니다."
    assert isinstance(result.score, float)

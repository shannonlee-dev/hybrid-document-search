"""BM25 원점수, 한국어 검색과 공통 검색 계약을 검증합니다."""

from math import isfinite, log
from pathlib import Path

import pytest

from data.loader import Document, load_prepared_documents
from retrievers.base import Retriever, SearchResult

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ko_miracl_prepared_corpus.jsonl"


@pytest.fixture
def retriever_class():
    pytest.importorskip("bm25s")
    pytest.importorskip("sklearn")
    from retrievers.bm25 import BM25Retriever

    return BM25Retriever


@pytest.fixture
def retriever(retriever_class):
    return retriever_class(load_prepared_documents(FIXTURE_PATH))


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
    result = retriever.search(query, top_k=1)[0]
    assert result.document_id == expected_id
    assert result.rank == 1
    assert isfinite(result.score) and result.score > 0


def test_top_k_ranking_and_metadata(retriever):
    shared: Retriever = retriever
    results = shared.search("대한민국", top_k=2)
    documents = {doc.document_id: doc for doc in load_prepared_documents(FIXTURE_PATH)}

    assert len(results) == 2
    assert [result.rank for result in results] == [1, 2]
    assert results[0].score >= results[1].score > 0
    assert len({result.document_id for result in results}) == 2
    for result in results:
        document = documents[result.document_id]
        assert isinstance(result, SearchResult)
        assert isinstance(result.score, float)
        assert result.title == document.title
        assert result.snippet == document.text[:200]


def test_top_k_can_exceed_corpus_size(retriever):
    results = retriever.search("대한민국", top_k=100)
    assert 0 < len(results) <= 4
    assert [result.rank for result in results] == list(range(1, len(results) + 1))
    assert all(result.score > 0 for result in results)


@pytest.mark.parametrize("query", ["", "   ", "\n\t", "zzzzzz"])
def test_empty_or_unknown_query(retriever, query):
    assert retriever.search(query, top_k=3) == []


@pytest.mark.parametrize("top_k", [0, -1, 1.5, "2", None, True])
def test_invalid_top_k(retriever, top_k):
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("서울", top_k=top_k)


@pytest.mark.parametrize("query", [None, 123])
def test_invalid_query_type(retriever, query):
    with pytest.raises(TypeError, match="query"):
        retriever.search(query, top_k=1)


def test_invalid_corpus(retriever_class):
    with pytest.raises(ValueError, match="문서"):
        retriever_class([])
    with pytest.raises(ValueError, match="중복 document_id"):
        retriever_class([Document("same#0", "서울"), Document("same#0", "부산")])
    with pytest.raises(ValueError, match="텍스트"):
        retriever_class([Document("empty#0", " \n\t ")])


@pytest.mark.parametrize("k1", [0, -1, float("nan"), float("inf"), True, "1.5"])
def test_invalid_k1(retriever_class, k1):
    with pytest.raises(ValueError, match="k1"):
        retriever_class([Document("doc#0", "서울")], k1=k1)


@pytest.mark.parametrize("b", [-0.1, 1.1, float("nan"), float("inf"), True, "0.75"])
def test_invalid_b(retriever_class, b):
    with pytest.raises(ValueError, match="b"):
        retriever_class([Document("doc#0", "서울")], b=b)


def test_ties_follow_corpus_order_and_search_is_repeatable(retriever_class):
    documents = (Document(id_, "동일한 본문") for id_ in ["z#0", "a#0"])
    retriever = retriever_class(documents)
    results = retriever.search("동일한", top_k=2)

    assert [result.document_id for result in results] == ["z#0", "a#0"]
    assert results[0].score == pytest.approx(results[1].score)
    assert results == retriever.search("동일한", top_k=2)


def test_title_search_preserves_original_body(retriever_class):
    text = "  원본 공백\n둘째 줄  " + "본문" * 120
    document = Document("doc#12", text, "서울")
    retriever = retriever_class([document, Document("other#0", "다른 내용")])
    result = retriever.search("서울", top_k=1)[0]

    assert result.document_id == "doc#12"
    assert result.title == "서울"
    assert result.snippet == text[:200]
    assert document.text == text


def test_zero_score_documents_are_excluded(retriever_class):
    retriever = retriever_class([Document("ko#0", "서울"), Document("en#0", "zzzzzz")])
    assert [r.document_id for r in retriever.search("서울", top_k=10)] == ["ko#0"]


def test_query_and_document_use_same_case_and_whitespace_rules(retriever_class):
    retriever = retriever_class([Document("doc#0", "SEOUL   대한민국")])
    assert retriever.search("seoul 대한민국", 1) == retriever.search(
        "SEOUL\n대한민국", 1
    )


@pytest.mark.parametrize("b", [0, 0.75, 1])
def test_raw_scores_match_hand_calculated_bm25(retriever_class, b):
    # 각 두 글자 단어는 char_wb 2~4-gram 6개를 생성합니다.
    # 문서 길이: 12, 6, 6 / 평균: 8 / 서울의 각 토큰 df: 2.
    k1 = 1.5
    retriever = retriever_class(
        [
            Document("twice#0", "서울 서울"),
            Document("once#0", "서울"),
            Document("no#0", "부산"),
        ],
        k1=k1,
        b=b,
    )
    results = retriever.search("서울", top_k=3)
    idf = log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    expected = {
        "twice#0": 6 * idf * 2 / (2 + k1 * (1 - b + b * 12 / 8)),
        "once#0": 6 * idf / (1 + k1 * (1 - b + b * 6 / 8)),
    }

    assert {result.document_id for result in results} == set(expected)
    for result in results:
        assert result.score == pytest.approx(expected[result.document_id], rel=1e-6)


def test_length_normalization_penalizes_longer_document(retriever_class):
    documents = [
        Document("long#0", "서울 " + "부산 " * 10),
        Document("short#0", "서울"),
    ]
    with_normalization = retriever_class(documents).search("서울", 2)
    without_normalization = retriever_class(documents, b=0).search("서울", 2)

    assert [r.document_id for r in with_normalization] == ["short#0", "long#0"]
    assert with_normalization[0].score > with_normalization[1].score
    assert without_normalization[0].score == pytest.approx(
        without_normalization[1].score
    )


def test_unknown_tokens_do_not_change_known_token_scores(retriever_class):
    retriever = retriever_class(
        [Document("doc#0", "서울"), Document("other#0", "부산")]
    )
    assert retriever.search("서울 zzzzzz", 2) == retriever.search("서울", 2)

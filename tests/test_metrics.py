"""합성 fixture의 손계산 값으로 공통 검색 평가 지표를 검증합니다."""

import json
from dataclasses import replace
from math import log2
from pathlib import Path

import pytest

from evaluation.metrics import (
    METRIC_NAMES,
    evaluate_query,
    evaluate_run,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from retrievers.base import SearchResult

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "evaluation_cases.json"
METRIC_FUNCTIONS = (recall_at_k, reciprocal_rank_at_k, ndcg_at_k)


@pytest.fixture
def evaluation_fixture():
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    run = {
        query_id: [SearchResult(**record) for record in records]
        for query_id, records in fixture["run"].items()
    }
    return run, fixture["qrels"]


def test_query_metrics_match_hand_calculated_values(evaluation_fixture):
    run, qrels = evaluation_fixture
    # 3개 관련 문서 중 2개를 순위 2, 4에서 찾습니다. 미판정과 라벨 0은 gain 0입니다.
    metrics = evaluate_query(run["mixed"], qrels["mixed"])
    assert metrics == pytest.approx(
        {
            "recall@5": 2 / 3,
            "recall@10": 2 / 3,
            "rr@10": 0.5,
            "ndcg@10": 0.49818925746641285,
        }
    )


def test_cutoff_includes_tenth_result_and_excludes_eleventh(evaluation_fixture):
    run, qrels = evaluation_fixture
    assert evaluate_query(run["late"], qrels["late"]) == {
        "recall@5": 0,
        "recall@10": 0,
        "rr@10": 0,
        "ndcg@10": 0,
    }
    assert evaluate_query(run["tenth"], qrels["tenth"]) == pytest.approx(
        {"recall@5": 0, "recall@10": 1, "rr@10": 0.1, "ndcg@10": 1 / log2(11)}
    )


def test_recall_cutoff_and_full_qrels_denominator():
    results = [SearchResult(f"doc-{rank}#0", rank, 1) for rank in range(1, 7)]
    qrels = {"doc-5#0": 1, "doc-6#0": 1, "not-retrieved#0": 1}
    assert recall_at_k(results, qrels, 5) == pytest.approx(1 / 3)
    assert recall_at_k(results, qrels, 6) == pytest.approx(2 / 3)


def test_reciprocal_rank_uses_only_first_relevant_result(evaluation_fixture):
    run, qrels = evaluation_fixture
    assert reciprocal_rank_at_k(run["mixed"], qrels["mixed"], 1) == 0
    assert reciprocal_rank_at_k(run["mixed"], qrels["mixed"], 2) == 0.5
    assert reciprocal_rank_at_k(run["mixed"], qrels["mixed"], 4) == 0.5


def test_run_macro_average_includes_missing_and_zero_positive_queries(
    evaluation_fixture,
):
    run, qrels = evaluation_fixture
    metrics = evaluate_run(run, qrels)
    assert tuple(metrics) == METRIC_NAMES
    assert metrics == pytest.approx(
        {
            "recall@5": 2 / 15,
            "recall@10": 1 / 3,
            "mrr@10": 0.12,
            "ndcg@10": (0.49818925746641285 + 1 / log2(11)) / 5,
        }
    )


@pytest.mark.parametrize("function", METRIC_FUNCTIONS)
def test_empty_results_and_no_relevant_documents_are_zero(function):
    assert function([], {"a#0": 1}, 10) == 0
    assert function([SearchResult("a#0", 1, 1)], {}, 10) == 0
    assert function([SearchResult("a#0", 1, 1)], {"a#0": 0}, 10) == 0


def test_positive_relevance_is_binary_gain():
    results = [SearchResult("a#0", 1, 1), SearchResult("b#0", 2, 0.5)]
    assert evaluate_query(results, {"a#0": 0.1, "b#0": 4}) == evaluate_query(
        results, {"a#0": 1, "b#0": 1}
    )
    assert ndcg_at_k(results, {"a#0": 0.1, "b#0": 4}, 10) == 1


def test_nonpositive_labels_and_unjudged_results_do_not_add_gain():
    results = [SearchResult("a#0", 1, 1), SearchResult("unknown#0", 2, 0.5)]
    assert evaluate_query(results, {"a#0": -1, "positive#0": 1}) == {
        "recall@5": 0,
        "recall@10": 0,
        "rr@10": 0,
        "ndcg@10": 0,
    }


def test_scores_do_not_change_ranking_or_mutate_results(evaluation_fixture):
    run, qrels = evaluation_fixture
    results = run["mixed"]
    original = list(results)
    rescored = [replace(result, score=-result.score) for result in results]
    assert evaluate_query(results, qrels["mixed"]) == evaluate_query(
        rescored, qrels["mixed"]
    )
    assert results == original


def test_recall_at_100_is_available_as_reference_only(evaluation_fixture):
    run, qrels = evaluation_fixture
    assert recall_at_k(run["late"], qrels["late"], 100) == 1
    assert "recall@100" not in evaluate_run(run, qrels)


def test_perfect_short_result_list_and_generator():
    qrels = {"a#0": 1, "b#0": 1}
    results = [SearchResult("a#0", 1, 1), SearchResult("b#0", 2, 0.5)]
    assert evaluate_query(iter(results), qrels) == {
        "recall@5": 1,
        "recall@10": 1,
        "rr@10": 1,
        "ndcg@10": 1,
    }


def test_ndcg_ideal_ranking_is_capped_at_k():
    qrels = {f"doc-{rank}#0": 1 for rank in range(1, 13)}
    results = [SearchResult(f"doc-{rank}#0", rank, 1) for rank in range(1, 11)]
    assert ndcg_at_k(results, qrels, 10) == 1
    assert recall_at_k(results, qrels, 10) == pytest.approx(10 / 12)


@pytest.mark.parametrize("function", METRIC_FUNCTIONS)
@pytest.mark.parametrize("k", [0, -1, 1.5, "10", None, True])
def test_invalid_cutoff(function, k):
    with pytest.raises(ValueError, match="k"):
        function([], {}, k)


@pytest.mark.parametrize("relevance", [True, None, "1", float("nan"), float("inf")])
def test_invalid_relevance(relevance):
    with pytest.raises(ValueError, match="relevance"):
        evaluate_query([], {"a#0": relevance})


@pytest.mark.parametrize("document_id", ["", "  ", 123])
def test_invalid_qrel_document_id(document_id):
    with pytest.raises(ValueError, match="document_id"):
        evaluate_query([], {document_id: 1})


def test_duplicate_results_are_rejected_even_beyond_cutoff():
    results = [SearchResult("a#0", 1, 1), SearchResult("a#0", 2, 0.5)]
    with pytest.raises(ValueError, match="중복"):
        recall_at_k(results, {"a#0": 1}, 1)


@pytest.mark.parametrize("rank", [0, 2, True, 1.0])
def test_invalid_rank(rank):
    with pytest.raises(ValueError, match="rank"):
        evaluate_query([SearchResult("a#0", rank, 1)], {"a#0": 1})


def test_out_of_order_rank_is_rejected():
    with pytest.raises(ValueError, match="rank"):
        evaluate_query([SearchResult("a#0", 2, 1), SearchResult("b#0", 1, 0.5)], {})


def test_invalid_result_type_and_id():
    with pytest.raises(TypeError, match="SearchResult"):
        evaluate_query(["a#0"], {})
    with pytest.raises(ValueError, match="document_id"):
        evaluate_query([SearchResult("", 1, 1)], {})


def test_invalid_qrels_mapping():
    with pytest.raises(TypeError, match="mapping"):
        evaluate_query([], [])


def test_empty_evaluation_set_is_rejected():
    with pytest.raises(ValueError, match="질의"):
        evaluate_run({}, {})


def test_all_missing_results_are_zero():
    assert evaluate_run({}, {"q1": {"a#0": 1}}) == dict.fromkeys(METRIC_NAMES, 0)


def test_unknown_query_is_rejected():
    with pytest.raises(ValueError, match="query_id"):
        evaluate_run({"extra": []}, {"q1": {"a#0": 1}})


@pytest.mark.parametrize("query_id", ["", " ", 123])
def test_invalid_query_id(query_id):
    with pytest.raises(ValueError, match="query_id"):
        evaluate_run({}, {query_id: {"a#0": 1}})


@pytest.mark.parametrize(("run", "qrels"), [([], {}), ({}, [])])
def test_invalid_run_mapping(run, qrels):
    with pytest.raises(TypeError, match="mapping"):
        evaluate_run(run, qrels)

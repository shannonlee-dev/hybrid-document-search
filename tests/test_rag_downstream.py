"""Verify downstream RAG helpers: fixed subsets, prompts, context metrics, stats."""

import json
from math import inf, nan

import pytest

from data.loader import Document
from evaluation import rag_downstream as rag
from retrievers.base import SearchResult
from scripts import evaluate_rag_downstream as cli


def test_smaller_subsets_are_prefixes_of_larger_ones():
    ids = [f"q{i:03d}_0" for i in range(200)]
    first = rag.shuffled_query_ids(ids, seed=7)
    assert first == rag.shuffled_query_ids(list(reversed(ids)), seed=7)
    assert sorted(first) == sorted(ids)
    assert first[:50] == rag.shuffled_query_ids(ids, seed=7)[:50]
    assert first != rag.shuffled_query_ids(ids, seed=8)
    with pytest.raises(ValueError):
        rag.shuffled_query_ids(["a", "a"])


def test_parent_id_groups_strategyqa_sub_questions():
    assert rag.parent_id("01c3faf4915a44133f60_1") == "01c3faf4915a44133f60"


def test_contexts_keep_rank_order_and_prompt_hides_method():
    documents = {
        "a": Document("a", "본문 A", "Title A"),
        "b": Document("b", "본문 B", None),
    }
    hits = [SearchResult("b", 1, 0.9), SearchResult("a", 2, 0.5)]
    contexts = rag.context_records(hits, documents, top_k=5)
    assert [c["document_id"] for c in contexts] == ["b", "a"]
    assert rag.context_text(contexts[1]) == "Title A\n본문 A"
    prompt = rag.build_prompt("질문?", contexts)
    assert prompt.index("[문서 1] 본문 B") < prompt.index("[문서 2] Title A")
    assert prompt.endswith("[Question]\n질문?\n\n[Answer]\n")
    for method in rag.RAG_METHODS:
        assert method not in prompt


def test_id_context_metrics_use_positive_qrels_only():
    contexts = [{"document_id": d} for d in ("a", "x", "b")]
    metrics = rag.id_context_metrics(contexts, {"a": 1, "b": 1, "c": 1, "x": 0})
    assert metrics == {
        "id_context_precision": pytest.approx(2 / 3),
        "id_context_recall": pytest.approx(2 / 3),
    }
    with pytest.raises(ValueError):
        rag.id_context_metrics(contexts, {"x": 0})


def test_abstention_flag_and_finite_mean():
    assert rag.is_abstention("문서에서 확인할 수 없습니다.")
    assert rag.is_abstention("알 수 없습니다")
    assert not rag.is_abstention("서울입니다.")
    assert rag.finite_mean([1.0, nan, None, 0.0]) == {
        "mean": 0.5,
        "n": 2,
        "undefined": 2,
    }


def test_paired_bootstrap_is_seeded_and_skips_undefined_pairs():
    pairs = [("g1", 1.0, 0.0), ("g1", 1.0, 1.0), ("g2", 0.5, 0.0), ("g3", nan, 0)]
    first = rag.paired_bootstrap(pairs, resamples=500, seed=1)
    assert first == rag.paired_bootstrap(pairs, resamples=500, seed=1)
    assert first["n"] == 3 and first["groups"] == 2
    assert first["mean_delta"] == pytest.approx(0.5)
    assert first["ci"][0] <= first["mean_delta"] <= first["ci"][1]
    assert (first["better"], first["worse"], first["tied"]) == (2, 0, 1)
    assert rag.paired_bootstrap([("g", None, 1.0)])["n"] == 0


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _fixture_rows(query_ids=("p1_0", "p2_0")):
    retrieval, generations, judged = [], [], []
    for position, query_id in enumerate(query_ids):
        for method in rag.RAG_METHODS:
            ids = ["a", "b"] if method != "score_fusion" or position else ["b", "a"]
            retrieval.append(
                {
                    "position": position,
                    "query_id": query_id,
                    "parent_id": rag.parent_id(query_id),
                    "method": method,
                    "question": "질문",
                    "contexts": [{"document_id": d} for d in ids],
                    "ir": {"recall@5": 1.0, "rr@10": 1.0, "ndcg@10": 1.0},
                    "id_context_precision": 0.5,
                    "id_context_recall": 1.0,
                }
            )
            generations.append(
                {
                    "method": method,
                    "query_id": query_id,
                    "response": "답",
                    "error": None,
                    "finish_reason": "length"
                    if (method, query_id) == ("rrf", "p2_0")
                    else "stop",
                }
            )
            score = 1.0 if method == "score_fusion" else 0.5
            judged.append(
                {
                    "method": method,
                    "query_id": query_id,
                    **{name: score for name in cli.RAGAS_METRICS},
                    **{f"{name}_error": None for name in cli.RAGAS_METRICS},
                }
            )
    return {
        "queries": [
            {"position": position, "query_id": query_id}
            for position, query_id in enumerate(query_ids)
        ],
        "retrieval": retrieval,
        "generations": generations,
        "ragas-results": judged,
    }


def _write_fixture(path, rows):
    for name, values in rows.items():
        # 기존 실행이 남긴 NaN/무한대를 재현하기 위해 json 기본값(allow_nan)으로 쓴다.
        _write_jsonl(path / f"{name}.jsonl", values)
    for name in ("config", "generation-config", "judge-config"):
        (path / f"{name}.json").write_text("{}", encoding="utf-8")


def _summarize(path, queries=2):
    cli.main(["summarize", "--output-dir", str(path), "--queries", str(queries)])
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def test_summarize_pairs_methods_on_identical_queries(tmp_path):
    _write_fixture(tmp_path, _fixture_rows())

    summary = _summarize(tmp_path)
    assert summary["queries"] == 2
    fusion = summary["paired"]["score_fusion-dense"]
    assert fusion["identical_top5_queries"] == 1
    assert fusion["same_set_different_order_queries"] == 1
    assert fusion["different_set_queries"] == 0
    assert fusion["metrics"]["faithfulness"]["mean_delta"] == pytest.approx(0.5)
    assert fusion["metrics"]["ndcg@10"]["mean_delta"] == 0
    assert summary["methods"]["dense"]["faithfulness"]["mean"] == 0.5
    # 잘린 생성은 본 결과에 남기고, 해당 비교에만 sensitivity를 덧붙인다.
    assert summary["truncated_generations"] == [
        {"method": "rrf", "query_id": "p2_0", "position": 1}
    ]
    assert "sensitivity_without_truncated" not in fusion
    rrf = summary["paired"]["rrf-dense"]
    assert rrf["metrics"]["faithfulness"]["n"] == 2
    sensitivity = rrf["sensitivity_without_truncated"]
    assert sensitivity["excluded_query_ids"] == ["p2_0"]
    assert sensitivity["metrics"]["faithfulness"]["n"] == 1
    assert (tmp_path / "ragas-results.csv").read_text(encoding="utf-8").count(
        "\n"
    ) == 11


def test_summarize_rejects_unfinished_rows(tmp_path):
    _write_jsonl(
        tmp_path / "retrieval.jsonl",
        [{"position": 0, "query_id": "q_0", "method": "dense"}],
    )
    with pytest.raises(SystemExit):
        cli.main(["summarize", "--output-dir", str(tmp_path), "--queries", "1"])


def _row(rows, name, method, query_id):
    return next(
        r for r in rows[name] if (r["method"], r["query_id"]) == (method, query_id)
    )


def _rejected(path, capsys, queries=2):
    with pytest.raises(SystemExit):
        cli.main(["summarize", "--output-dir", str(path), "--queries", str(queries)])
    assert not (path / "summary.json").exists()
    return capsys.readouterr().err


def test_summarize_rejects_substituted_query_with_equal_row_counts(tmp_path, capsys):
    # 리뷰 재현: 행 수는 같지만 BM25의 두 번째 질의만 다른 ID다.
    rows = _fixture_rows()
    for name in ("retrieval", "generations", "ragas-results"):
        _row(rows, name, "bm25", "p2_0")["query_id"] = "p3_0"
    _write_fixture(tmp_path, rows)
    error = _rejected(tmp_path, capsys)
    assert "bm25: missing=['p2_0'], unexpected=['p3_0']" in error
    assert "dense:" not in error


@pytest.mark.parametrize("name", ["retrieval", "generations", "ragas-results"])
def test_summarize_rejects_duplicate_method_query_rows(tmp_path, capsys, name):
    rows = _fixture_rows()
    rows[name].append(dict(_row(rows, name, "dense", "p1_0")))
    _write_fixture(tmp_path, rows)
    error = _rejected(tmp_path, capsys)
    assert f"{name}.jsonl에 중복 (method, query_id)" in error
    assert "('dense', 'p1_0')" in error


def test_summarize_rejects_missing_query_and_position_mismatch(tmp_path, capsys):
    rows = _fixture_rows()
    rows["retrieval"].remove(_row(rows, "retrieval", "tfidf", "p2_0"))
    _row(rows, "retrieval", "dense", "p1_0")["position"] = 1
    _row(rows, "retrieval", "dense", "p2_0")["position"] = 0
    _write_fixture(tmp_path, rows)
    error = _rejected(tmp_path, capsys)
    assert "tfidf: missing=['p2_0']" in error
    assert "dense: position_mismatch=['p1_0@1!=0', 'p2_0@0!=1']" in error


def test_summarize_rejects_unknown_generation_rows(tmp_path, capsys):
    rows = _fixture_rows()
    _row(rows, "generations", "rrf", "p1_0")["query_id"] = "p9_0"
    _write_fixture(tmp_path, rows)
    assert "generations.jsonl에 retrieval.jsonl에 없는 행" in _rejected(
        tmp_path, capsys
    )


def test_summarize_requires_the_frozen_query_prefix(tmp_path, capsys):
    _write_fixture(tmp_path, _fixture_rows())
    assert "요청한 3개 질의가 없습니다: 2개" in _rejected(tmp_path, capsys, queries=3)
    (tmp_path / "queries.jsonl").unlink()
    assert "요청한 2개 질의가 없습니다: 0개" in _rejected(tmp_path, capsys)


def test_summarize_accepts_a_prefix_of_finished_rows(tmp_path):
    _write_fixture(tmp_path, _fixture_rows())
    summary = _summarize(tmp_path, queries=1)
    assert summary["queries"] == 1
    assert summary["methods"]["dense"]["faithfulness"]["n"] == 1


def _strict_json(text):
    def reject(constant):
        raise AssertionError(f"non-standard JSON constant: {constant}")

    return json.loads(text, parse_constant=reject)


@pytest.mark.parametrize("bad", [nan, inf, -inf])
def test_summarize_treats_non_finite_metrics_as_undefined(tmp_path, capsys, bad):
    rows = _fixture_rows()
    # score_fusion·dense는 p2_0에서 같은 context이므로 noise 추정에서도 빠져야 한다.
    _row(rows, "ragas-results", "score_fusion", "p2_0")["faithfulness"] = bad
    # rrf p2_0은 dense와 다른 context가 되어 differing_queries의 delta를 거친다.
    _row(rows, "retrieval", "rrf", "p2_0")["contexts"] = [{"document_id": "c"}]
    _row(rows, "ragas-results", "rrf", "p2_0")["faithfulness"] = bad
    _row(rows, "retrieval", "bm25", "p1_0")["ir"]["ndcg@10"] = bad
    _write_fixture(tmp_path, rows)

    summary = _summarize(tmp_path)
    text = (tmp_path / "summary.json").read_text(encoding="utf-8")
    assert _strict_json(text) == summary
    _strict_json(capsys.readouterr().out)

    fusion = summary["methods"]["score_fusion"]
    assert fusion["faithfulness"] == {"mean": 1.0, "n": 1, "undefined": 1}
    assert fusion["judge_errors"]["faithfulness"] == 1
    assert fusion["context_precision"] == {"mean": 1.0, "n": 2, "undefined": 0}
    assert summary["methods"]["bm25"]["ndcg@10"] == {
        "mean": 1.0,
        "n": 1,
        "undefined": 1,
    }
    assert summary["methods"]["dense"]["faithfulness"]["n"] == 2

    paired = summary["paired"]["score_fusion-dense"]["metrics"]
    assert paired["faithfulness"]["n"] == 1
    assert paired["faithfulness"]["mean_delta"] == pytest.approx(0.5)
    assert paired["faithfulness"]["ci"] == [pytest.approx(0.5)] * 2
    assert paired["faithfulness"]["identical_context_mean_abs_delta"] is None
    assert paired["context_precision"]["n"] == 2
    assert paired["context_precision"]["identical_context_mean_abs_delta"] == 0.5

    rrf = summary["paired"]["rrf-dense"]
    assert rrf["metrics"]["faithfulness"]["n"] == 1
    (differing,) = rrf["differing_queries"]
    assert differing["query_id"] == "p2_0"
    assert differing["deltas"]["faithfulness"] is None
    assert differing["deltas"]["context_precision"] == 0.0

    csv_text = (tmp_path / "ragas-results.csv").read_text(encoding="utf-8")
    assert cli.NON_FINITE in csv_text
    assert "nan" not in csv_text.lower() and "inf" not in csv_text.lower()


def test_non_finite_metric_values_become_undefined_with_a_reason(tmp_path):
    for bad in (nan, inf, -inf, None):
        assert rag.finite_or_none(bad) is None
    assert rag.finite_or_none(0.0) == 0.0
    assert cli._metric_value(nan, None) == (None, cli.NON_FINITE)
    assert cli._metric_value(-inf, "TimeoutError: x") == (None, "TimeoutError: x")
    assert cli._metric_value(0.25, None) == (0.25, None)
    with pytest.raises(ValueError):
        cli._append_jsonl(tmp_path / "rows.jsonl", {"faithfulness": nan})

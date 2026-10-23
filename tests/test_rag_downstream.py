"""Verify downstream RAG helpers: fixed subsets, prompts, context metrics, stats."""

import json
from math import nan

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


def test_summarize_pairs_methods_on_identical_queries(tmp_path):
    retrieval, generations, judged = [], [], []
    for position, query_id in enumerate(("p1_0", "p2_0")):
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
    _write_jsonl(tmp_path / "retrieval.jsonl", retrieval)
    _write_jsonl(tmp_path / "generations.jsonl", generations)
    _write_jsonl(tmp_path / "ragas-results.jsonl", judged)
    for name in ("config", "generation-config", "judge-config"):
        (tmp_path / f"{name}.json").write_text("{}", encoding="utf-8")

    cli.main(["summarize", "--output-dir", str(tmp_path), "--queries", "2"])

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
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

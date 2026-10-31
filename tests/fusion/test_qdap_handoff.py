"""Pin query identity, metric meaning and prediction alignment in handoff rows."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")


def test_handoff_maps_ids_and_exports_real_metrics_alpha_and_missed_relevance():
    from evaluation.qdap_handoff import per_query_records

    records = per_query_records(
        ["query-b", "query-a"],
        ["doc-z", "doc-a", "doc-m"],
        {"QDAP-S": np.array([[1, 0], [0, 1]])},
        [{0: 3}, {2: 3}],
        run_id="frozen-run",
        alphas={"QDAP-S": [0.73, 0.12]},
        oracle_alphas=[0.9, 0.4],
    )
    first, second = records
    assert first["query_id"] == "query-b"
    assert first["run_id"] == "frozen-run"
    assert first["top10_document_ids"] == ["doc-a", "doc-z"]
    assert first["p@1"] == 0
    assert first["recall@10"] == 1
    assert first["mrr@10"] == 0.5
    assert first["ndcg@10"] == pytest.approx(0.6309297535714575)
    assert first["relevant_rank_top10"] == 2
    assert first["selected_alpha_dense"] == 0.73
    assert second["query_id"] == "query-a"
    assert second["selected_alpha_dense"] == 0.12
    assert second["oracle_alpha_dense"] == 0.4
    assert second["relevant_rank_top10"] is None
    assert second["ndcg@10"] == 0


def test_handoff_rejects_duplicate_queries_padding_and_misaligned_predictions():
    from evaluation.qdap_handoff import per_query_records

    with pytest.raises(ValueError, match="unique"):
        per_query_records(
            ["q", "q"], ["d"], {"S": np.array([[0], [0]])}, [{0: 1}, {0: 1}], run_id="r"
        )
    with pytest.raises(ValueError, match="document"):
        per_query_records(["q"], ["d"], {"S": np.array([[-1]])}, [{0: 1}], run_id="r")
    with pytest.raises(ValueError, match="alpha"):
        per_query_records(
            ["q"], ["d"], {"S": np.array([[0]])}, [{0: 1}], run_id="r", alphas={"S": []}
        )


def test_handoff_keeps_methods_and_query_order_separate():
    from evaluation.qdap_handoff import per_query_records

    records = per_query_records(
        ["q"],
        ["a", "b"],
        {"Fixed": np.array([[1, 0]]), "S": np.array([[0, 1]])},
        [{0: 1}],
        run_id="r",
        alphas={"S": [0.8]},
        modes={"Fixed": ["balanced"]},
    )
    assert [(r["method"], r["p@1"]) for r in records] == [("Fixed", 0), ("S", 1)]
    assert records[0]["selected_alpha_dense"] is None
    assert records[0]["selected_mode"] == "balanced"
    assert records[1]["selected_alpha_dense"] == 0.8

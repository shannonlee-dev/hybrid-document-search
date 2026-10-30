"""QDAP score preparation uses only the shared corpus, BM25 and frozen vectors."""

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("bm25s")


def test_prepare_needs_no_external_score_cache(score_inputs):
    from evaluation.qdap_benchmark import prepare

    out = score_inputs / "run"
    prepare(score_inputs, out)
    for split in ("train", "validation"):
        with np.load(out / f"{split}.npz") as saved:
            assert saved["pools"].shape == (2, 2)
            np.testing.assert_array_equal(saved["quality"][:, -1], [1, 1])
    assert not (score_inputs / "artifacts/paper-guided").exists()


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_all_splits_compute_aligned_float64_dense_and_real_bm25(score_inputs, split):
    from data.loader import Document
    from evaluation.qdap_benchmark import rows, split_scores
    from retrievers.bm25 import BM25Retriever

    data = score_inputs / "data/processed/bccard"
    queries = rows(data / f"queries_{split}.jsonl")
    dense, sparse = split_scores(score_inputs, split, queries[::-1])
    np.testing.assert_array_equal(dense, [[0, 1], [1, 0]])
    assert dense.dtype == sparse.dtype == np.float64
    bm25 = BM25Retriever([Document(**d) for d in rows(data / "corpus.jsonl")])
    np.testing.assert_array_equal(
        sparse, [bm25.score_all("분실 신고"), bm25.score_all("카드 결제")]
    )


def test_score_generation_rejects_vector_query_order_mismatch(score_inputs):
    from evaluation.qdap_benchmark import rows, split_scores

    cache = score_inputs / "artifacts/bccard-adaptive-vectors-resumed/encoding.json"
    record = json.loads(cache.read_text())
    record["splits"]["test"]["query_ids"].reverse()
    cache.write_text(json.dumps(record))
    queries = rows(score_inputs / "data/processed/bccard/queries_test.jsonl")
    with pytest.raises(ValueError, match="order"):
        split_scores(score_inputs, "test", queries)

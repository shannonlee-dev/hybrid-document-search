"""QDAP score preparation uses only the shared corpus, BM25 and frozen vectors."""

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("bm25s")


def test_prepare_needs_no_external_score_cache(score_inputs):
    from evaluation.json_io import read_json, write_json
    from evaluation.qdap_artifacts import sha
    from evaluation.qdap_benchmark import prepare

    data = score_inputs / "data/processed/bccard"
    manifest_path = data / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["split_stats"] = {
        "train": {
            "qrels_source": "reconstructed with the dataset-card chunk-distance rule"
        },
        "validation": {
            "qrels_source": "official relevant_chunk_ids / relevance_scores"
        },
        "test": {"qrels_source": "official relevant_chunk_ids / relevance_scores"},
    }
    write_json(manifest_path, manifest)
    encoding_path = (
        score_inputs / "artifacts/bccard-adaptive-vectors-resumed/encoding.json"
    )
    encoding = read_json(encoding_path)
    encoding["dataset_manifest_sha256"] = sha(manifest_path)
    write_json(encoding_path, encoding)
    out = score_inputs / "run"
    prepare(score_inputs, out)
    protocol = read_json(out / "protocol.json")
    assert protocol["qrels_provenance"] == {
        split: stats["qrels_source"] for split, stats in manifest["split_stats"].items()
    }
    assert "Train qrels" in protocol["training_target"]
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


@pytest.mark.parametrize(
    "filename", ["manifest.json", "document-vectors.npy", "query-vectors-test.npy"]
)
def test_score_generation_rejects_changed_dataset_or_vectors(score_inputs, filename):
    from evaluation.qdap_benchmark import rows, split_scores

    data = score_inputs / "data/processed/bccard"
    queries = rows(data / "queries_test.jsonl")
    if filename == "manifest.json":
        path = data / filename
        path.write_text(path.read_text() + " ")
        message = "manifest mismatch"
    else:
        path = score_inputs / "artifacts/bccard-adaptive-vectors-resumed" / filename
        np.save(path, np.zeros((2, 2), dtype=np.float32))
        message = "vector hash mismatch"
    with pytest.raises(ValueError, match=message):
        split_scores(score_inputs, "test", queries)


def test_prepare_and_small_training_need_no_test_qrels_or_model(score_inputs):
    import shutil
    from types import SimpleNamespace

    import torch

    from evaluation.json_io import read_json
    from evaluation.qdap_benchmark import prepare
    from scripts.run_qdap_benchmark import train

    data = score_inputs / "data/processed/bccard"
    (data / "qrels_test.jsonl").unlink()
    cache = score_inputs / "artifacts/bccard-adaptive-vectors-resumed"
    (cache / "query-vectors-test.npy").unlink()
    external = score_inputs / "external-vectors"
    cache.rename(external)
    out = score_inputs / "run"
    prepare(score_inputs, out, external)
    shutil.rmtree(data)
    shutil.rmtree(external)
    torch.set_num_threads(2)
    args = SimpleNamespace(
        output=out, inputs=None, variant="S", ce_mode="literal", device="cpu"
    )
    train(args)
    for seed in (42, 43, 44):
        report = read_json(out / f"S-literal-{seed}/training.json")
        assert report["train_queries"] == 2
        assert len(report["history"]) == 5
    with pytest.raises(FileExistsError, match="already exists"):
        train(args)
    with pytest.raises(FileExistsError):
        prepare(score_inputs, out, external)

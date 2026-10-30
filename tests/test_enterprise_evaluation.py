"""가짜 embedding으로 BC Card baseline 평가, slice 집계와 test oracle 차단을 검증합니다."""

import hashlib

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("bm25s")
pytest.importorskip("faiss")

from data.bccard import prepare_bccard  # noqa: E402
from evaluation import enterprise  # noqa: E402
from evaluation.enterprise import (  # noqa: E402
    encode_bccard,
    preference_analysis,
    run_enterprise_baselines,
    slice_definitions,
)
from tests.test_bccard import _write  # noqa: E402


class _FakeModel:
    max_seq_length = 8192

    def parameters(self):
        yield np.zeros(1, dtype=np.float32)


class _FakeEmbedder:
    """Map each text to a deterministic unit vector without loading a model."""

    def __init__(self, config):
        self.config = config
        self.model = _FakeModel()

    def _vector(self, text):
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
        vector = np.random.default_rng(seed).normal(size=(1, 8)).astype(np.float32)
        return vector / np.linalg.norm(vector)

    def encode_documents(self, texts):
        return np.concatenate([self._vector(text) for text in texts])

    def encode_query(self, query):
        return self._vector(query)


@pytest.fixture
def encoded(tmp_path, monkeypatch):
    data = tmp_path / "prepared"
    prepare_bccard(*_write(tmp_path / "raw"), data)
    monkeypatch.setattr("retrievers.dense.DenseEmbedder", _FakeEmbedder)
    cache = tmp_path / "cache"
    record = encode_bccard(data, cache, device="cpu")
    return data, cache, record


def test_encoding_records_model_and_hashes(encoded):
    _, cache, record = encoded
    assert record["model"]["model_name"] == "BAAI/bge-m3"
    assert record["model"]["batch_size"] == 1
    assert set(record["splits"]) == {"validation", "test"}
    assert np.load(cache / "document-vectors.npy").shape == (10, 8)


def test_kanana_encoding_records_pinned_external_model(tmp_path, monkeypatch):
    data = tmp_path / "prepared"
    prepare_bccard(*_write(tmp_path / "raw"), data)
    monkeypatch.setattr("retrievers.dense.DenseEmbedder", _FakeEmbedder)
    record = encode_bccard(data, tmp_path / "cache", device="cpu", encoder="kanana")
    assert record["model"]["model_name"] == "kakaocorp/kanana-2-3b-embedding"
    assert record["model"]["revision"] == "2c85dbe61af8ff033d2c9d46a686c665153069aa"
    assert record["model"]["trust_remote_code"] is True
    assert record["model"]["query_prefix"] == record["model"]["passage_prefix"] == ""
    assert record["embedding_dimension"] == 8


def test_unknown_encoder_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="encoder"):
        encode_bccard(tmp_path, tmp_path / "cache", encoder="e5")


def test_baselines_report_slices_and_existing_metrics(encoded):
    data, cache, _ = encoded
    report = run_enterprise_baselines(data, cache, "test", warmup=1, repeats=1)
    assert list(report["methods"]) == ["bm25", "dense", "score_fusion"]
    assert report["methods"]["score_fusion"]["config"]["bm25_weight"] == 0.05
    summary = report["methods"]["bm25"]["slices"]
    assert summary["all"]["query_count"] == 2
    assert summary["non_synthetic"]["query_count"] == 2
    assert summary["text_not_in_train"]["query_count"] == 1
    for value in summary["all"].values():
        assert 0 <= value <= 2


def test_cache_from_other_manifest_is_rejected(encoded, tmp_path):
    data, cache, _ = encoded
    (data / "manifest.json").write_text(
        (data / "manifest.json").read_text() + " ", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="manifest"):
        run_enterprise_baselines(data, cache, "test", warmup=1, repeats=1)


def test_train_is_not_a_baseline_report_split(encoded):
    data, cache, _ = encoded
    with pytest.raises(ValueError, match="train"):
        run_enterprise_baselines(data, cache, "train")


def test_oracle_analysis_refuses_held_out_test(encoded):
    data, cache, _ = encoded
    report = run_enterprise_baselines(data, cache, "test", warmup=1, repeats=1)
    with pytest.raises(ValueError, match="held-out test"):
        preference_analysis(report, {"all": ["base_bcai_000003_000"]})


def _report(scores):
    return {
        "dataset": {"split": "validation"},
        "methods": {
            method: {
                "queries": [
                    {"query_id": query_id, "metrics": {"ndcg@10": value}}
                    for query_id, value in values.items()
                ]
            }
            for method, values in scores.items()
        },
    }


def test_preference_analysis_counts_wins_ties_and_oracle_gain():
    report = _report(
        {
            "bm25": {"a": 1.0, "b": 0.0, "c": 0.5},
            "dense": {"a": 0.0, "b": 1.0, "c": 0.5},
            "score_fusion": {"a": 0.5, "b": 1.0, "c": 0.5},
        }
    )
    result = preference_analysis(report, {"all": ["a", "b", "c"]})["slices"]["all"]
    assert result["bm25_gt_dense"] == pytest.approx(1 / 3)
    assert result["dense_gt_bm25"] == pytest.approx(1 / 3)
    assert result["bm25_dense_tied"] == pytest.approx(1 / 3)
    assert result["all_three_tied"] == pytest.approx(1 / 3)
    assert result["fusion_gt_both"] == 0
    assert result["fusion_lt_best_component"] == pytest.approx(1 / 3)
    assert result["best_fixed"] == "score_fusion"
    assert result["oracle_gain_over_best_fixed"] == pytest.approx(
        (1.0 + 1.0 + 0.5) / 3 - 2.0 / 3
    )


def test_slices_come_from_metadata_fields_only():
    metadata = {
        "q1": {"provenance": "base", "page_type": None, "site": None},
        "q2": {"provenance": "synthetic_crawl", "page_type": "faq", "site": "bc-vip"},
    }
    slices = slice_definitions(metadata)
    assert slices["non_synthetic"] == ["q1"] and slices["synthetic"] == ["q2"]
    assert slices["page_type=faq"] == ["q2"] and "text_not_in_train" not in slices
    assert enterprise.REPORT_METRICS[0] == "p@1"


def test_handoff_refuses_test_and_keeps_fusion_on_ties(encoded):
    from data.bccard import load_query_metadata
    from evaluation.data import load_evaluation_dataset
    from evaluation.enterprise import validation_handoff

    data, cache, _ = encoded
    for split in ("validation", "test"):
        report = run_enterprise_baselines(data, cache, split, warmup=1, repeats=1)
        dataset = load_evaluation_dataset(data, split)
        args = (
            report,
            load_query_metadata(data, split),
            dataset.qrels,
            dataset.documents,
        )
        if split == "test":
            with pytest.raises(ValueError, match="held-out test"):
                validation_handoff(*args)
            continue
        handoff = validation_handoff(*args)
        ties = handoff["wins"]["all_three_tied"]["count"]
        assert handoff["oracle"]["choice_counts"].get("score_fusion", 0) >= ties
        assert sum(handoff["oracle"]["choice_counts"].values()) == 2


def test_query_properties_are_observable_surface_features():
    from evaluation.enterprise import query_properties

    result = query_properties("PG사 수수료 2%는?", "PG사 결제 수수료")
    assert result["has_digit"] and result["has_latin"]
    assert result["has_uppercase_abbreviation"]
    assert 0 < result["bigram_overlap_with_positive"] < 1

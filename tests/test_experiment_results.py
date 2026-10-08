import copy

import pytest

from evaluation.benchmark import benchmark_retriever
from evaluation.data import load_evaluation_dataset
from retrievers.base import SearchResult


@pytest.fixture
def result_fixture(evaluation_directory):
    class Retriever:
        def search(self, query, top_k):
            return [SearchResult("fixture-001#0", 1, 1)]

    dataset = load_evaluation_dataset(evaluation_directory)
    return dataset, benchmark_retriever(Retriever(), dataset)


def test_quality_and_latency_recomputed(result_fixture):
    from evaluation.experiment_results import _validate_method

    dataset, result = result_fixture
    _validate_method(result, dataset)
    for mutation in ("metrics", "latency", "samples", "id", "text"):
        broken = copy.deepcopy(result)
        if mutation == "metrics":
            broken["metrics"]["ndcg@10"] += 0.1
        elif mutation == "latency":
            broken["latency"]["mean_ms"] += 1
        elif mutation == "samples":
            broken["queries"][0]["latency_samples_ms"].pop()
        elif mutation == "id":
            broken["queries"][0]["results"][0]["document_id"] = "unknown"
        else:
            broken["queries"][0]["text"] = "changed"
        with pytest.raises(ValueError):
            _validate_method(broken, dataset)


def test_revision_mismatch_fails():
    from evaluation.dense_runtime import MODELS
    from evaluation.experiment_results import _validate_embedding

    model, revision = next(iter(MODELS.items()))
    config = {
        "model_name": model,
        "revision": revision,
        "batch_size": 1,
        "normalize_embeddings": True,
        "device": "cuda:0",
    }
    _validate_embedding(config, model)
    config["revision"] = "0" * 40
    with pytest.raises(ValueError, match="revision"):
        _validate_embedding(config, model)


def test_selection_uses_configured_default_without_reason(tmp_path, monkeypatch):
    from evaluation import experiment_results
    from retrievers.model_config import ALIASES, MODELS

    default_model = ALIASES["bge"]
    monkeypatch.setattr(experiment_results, "DEFAULT_MODEL", default_model)
    rows = {
        "e5": {"metrics": {"ndcg@10": 0.9}},
        "bge": {"metrics": {"ndcg@10": 0.7}},
        "kure": {"metrics": {"ndcg@10": 0.8}},
    }
    path = tmp_path / "selection.json"
    experiment_results._write_selection(path, rows)
    selection = experiment_results.read_json(path)
    assert selection == {
        "selection_policy": "config_default",
        "model": default_model,
        "revision": MODELS[default_model],
        "train_ndcg_ranking": ["e5", "kure", "bge"],
    }

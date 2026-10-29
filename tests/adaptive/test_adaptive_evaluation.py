"""Adaptive 학습·동결·평가 흐름과 test 데이터 사용 제한 테스트."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("bm25s")
pytest.importorskip("faiss")

from data.bccard import prepare_bccard  # noqa: E402
from evaluation.enterprise import encode_bccard  # noqa: E402
from evaluation.json_io import read_json  # noqa: E402
from tests.test_bccard import _write  # noqa: E402
from tests.test_enterprise_evaluation import _FakeEmbedder  # noqa: E402


@pytest.mark.parametrize("empty_adaptive_results", [False, True])
def test_fit_freeze_evaluate_and_no_overwrite(
    tmp_path, monkeypatch, empty_adaptive_results
):
    # 학습·동결·평가를 수행하고, 산출물 변조·덮어쓰기·test 재실행을 거부한다.
    from evaluation.adaptive import evaluate_frozen, fit_router

    data = tmp_path / "data"
    prepare_bccard(*_write(tmp_path / "raw"), data)
    monkeypatch.setattr("retrievers.dense.DenseEmbedder", _FakeEmbedder)
    cache = tmp_path / "cache"
    encode_bccard(data, cache, splits=("train", "validation", "test"), device="cpu")
    output = tmp_path / "run"
    frozen = fit_router(data, cache, output)
    assert frozen["training_split"] == "train"
    assert frozen["selection_split"] == "validation"
    assert len(frozen["candidates"]) == 60
    assert "test" not in frozen["quality_inputs"]
    with pytest.raises(ValueError, match="exists"):
        fit_router(data, cache, output)
    # 모델이 변조되면 test 시작 marker를 만들기 전에 실패해야 한다.
    original = (output / "router.json").read_bytes()
    (output / "router.json").write_bytes(original + b" ")
    with pytest.raises(ValueError, match="changed"):
        evaluate_frozen(data, cache, output, warmup=1, repeats=1)
    assert not (output / "test-started.json").exists()
    (output / "router.json").write_bytes(original)
    from evaluation import benchmark

    digest = benchmark.file_sha256
    with monkeypatch.context() as patch:
        patch.setattr(
            benchmark,
            "file_sha256",
            lambda path: "changed" if path.name == "constants.py" else digest(path),
        )
        with pytest.raises(ValueError, match="implementation changed"):
            evaluate_frozen(data, cache, output, warmup=1, repeats=1)
        assert not (output / "test-started.json").exists()
    if empty_adaptive_results:
        from fusion.adaptive import AdaptiveRetriever

        original_search = AdaptiveRetriever.search_details

        def empty_search(self, query, top_k):
            _, details = original_search(self, query, top_k)
            return [], details

        monkeypatch.setattr(AdaptiveRetriever, "search_details", empty_search)
    report = evaluate_frozen(data, cache, output, warmup=1, repeats=1)
    assert set(report["methods"]) == {
        "bm25",
        "dense",
        "score_fusion",
        "adaptive",
        "oracle",
    }
    assert report["methods"]["adaptive"]["slices"]["non_synthetic"]["query_count"] == 2
    assert sum(report["mode_counts"].values()) == 2
    saved = read_json(output / "test-report.json")
    assert set(saved["ablation"]["methods"]) == {
        "dense",
        "score_fusion",
        "adaptive",
        "oracle",
    }
    delta = (
        saved["ablation"]["methods"]["adaptive"]["ndcg@10"]
        - saved["ablation"]["methods"]["score_fusion"]["ndcg@10"]
    )
    assert saved["ablation"]["adaptive_minus_fixed_ndcg@10"] == delta
    assert saved["ablation"]["outcome"] in {"improved", "degraded", "tied"}
    if empty_adaptive_results:
        assert saved["ablation"]["outcome"] == "degraded"
        assert delta < 0
        assert (
            read_json(output / "router.json")["weights"]
            == frozen["selected"]["weights"]
        )
    assert report["methods"]["oracle"]["latency"] is None
    assert (
        report["methods"]["oracle"]["slices"]["all"]["ndcg@10"]
        >= report["methods"]["adaptive"]["slices"]["all"]["ndcg@10"]
    )
    assert (
        read_json(output / "test-report.json")["router_sha256"]
        == frozen["router_sha256"]
    )
    with pytest.raises(ValueError, match="already"):
        evaluate_frozen(data, cache, output, warmup=1, repeats=1)


def test_test_not_used_for_candidate_collection():
    # 후보 선택용 데이터 수집에 test split을 사용하지 못하게 한다.
    from evaluation.adaptive import collect_training_table

    with pytest.raises(ValueError, match="train.*validation"):
        collect_training_table("missing", "missing", "test")


def test_validation_selects_weights_without_test_queries(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from evaluation import adaptive
    from evaluation.json_io import write_json

    config = tmp_path / "candidates.json"
    write_json(config, {"weight_sets": [[0.2, 0.05, 0], [0.3, 0.1, 0.01]]})
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "encoding.json").write_text("{}")
    calls = []

    def collect(
        data_dir, cache_dir, split, progress=None, *, weight_sets, table_path=None
    ):
        calls.append(split)
        assert split in ("train", "validation")
        weights = sorted({0.05, *(w for triple in weight_sets for w in triple)})
        # Both candidates learn balanced on train; validation prefers weight 0.1.
        quality = np.zeros((1, len(weights)))
        quality[0, weights.index(0.05)] = 0.4
        quality[0, weights.index(0.1)] = 0.8
        dataset = SimpleNamespace(
            split=split, provenance={}, queries=[SimpleNamespace(query_id="q")]
        )
        return dataset, np.zeros((1, 10)), quality, weights

    monkeypatch.setattr(adaptive, "collect_training_table", collect)
    monkeypatch.setattr(adaptive, "load_query_metadata", lambda *a: {})
    monkeypatch.setattr(
        adaptive, "slice_definitions", lambda *a: {"non_synthetic": ["q"]}
    )
    output = tmp_path / "run"
    frozen = adaptive.fit_router(tmp_path, tmp_path, output, candidates_path=config)
    assert calls == ["train", "validation"]
    assert frozen["selected"]["weights"] == [0.3, 0.1, 0.01]
    assert frozen["selected"]["classifier"] == "logistic"
    assert read_json(output / "train-labels.json") == [
        {"query_id": "q", "mode": "balanced"}
    ]
    assert read_json(output / "protocol.json")["weight_sets"] == [
        [0.2, 0.05, 0],
        [0.3, 0.1, 0.01],
    ]


@pytest.mark.parametrize(
    "weights", [[], [[0.1, 0.2, 0]], [[0.2, 0.05]], [[0.2, 0.05, -1]]]
)
def test_invalid_candidate_weights_rejected_before_run(tmp_path, weights):
    from evaluation.adaptive import fit_router
    from evaluation.json_io import write_json

    config = tmp_path / "candidates.json"
    write_json(config, {"weight_sets": weights})
    with pytest.raises(ValueError, match="weight"):
        fit_router("missing", "missing", tmp_path / "run", candidates_path=config)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("command", ["encode", "evaluate"])
def test_candidates_cannot_override_frozen_evaluation(command, capsys):
    from scripts.evaluate_adaptive import main

    with pytest.raises(SystemExit):
        main([command, "--candidates", "changed.json"])
    assert "only accepted for fit" in capsys.readouterr().err


def test_training_table_cache_reuses_scores_and_resumes(tmp_path, monkeypatch):
    from evaluation.adaptive import collect_training_table
    from evaluation.json_io import write_json
    from fusion.adaptive import AdaptiveRetriever

    data = tmp_path / "data"
    prepare_bccard(*_write(tmp_path / "raw"), data)
    monkeypatch.setattr("retrievers.dense.DenseEmbedder", _FakeEmbedder)
    cache = tmp_path / "vectors"
    encode_bccard(data, cache, splits=("train",), device="cpu")
    path = tmp_path / "table.json"
    weights = [[0.2, 0.05, 0]]
    calls = []
    original = AdaptiveRetriever.score_arrays

    def counted(self, query):
        calls.append(query)
        return original(self, query)

    monkeypatch.setattr(AdaptiveRetriever, "score_arrays", counted)
    dataset, x, quality, _ = collect_training_table(
        data, cache, "train", weight_sets=weights, table_path=path
    )
    count = len(dataset.queries)
    assert len(calls) == count
    _, cached_x, cached_quality, _ = collect_training_table(
        data, cache, "train", weight_sets=weights, table_path=path
    )
    assert len(calls) == count
    assert np.array_equal(cached_x, x)
    assert np.array_equal(cached_quality, quality)
    checkpoint = read_json(path)
    checkpoint["features"] = checkpoint["features"][:1]
    checkpoint["quality"] = checkpoint["quality"][:1]
    write_json(path, checkpoint)
    collect_training_table(data, cache, "train", weight_sets=weights, table_path=path)
    assert len(calls) == count + count - 1
    with pytest.raises(ValueError, match="cache.*match"):
        collect_training_table(
            data, cache, "train", weight_sets=[[0.4, 0.05, 0]], table_path=path
        )

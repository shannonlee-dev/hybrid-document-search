"""Adaptive feature, 모드 선택, 모델 저장·복원 및 검색 결과 테스트."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("faiss")
pytest.importorskip("bm25s")

from tests.fusion.test_score_fusion_retriever import sources  # noqa: E402, F401


def test_features_handle_short_corpus_and_no_sparse_signal():
    # 짧거나 빈 corpus, BM25 신호 부재, NaN 입력의 feature 처리를 확인한다.
    from fusion.adaptive import query_features

    values = query_features("가A2 ", [0], [0.8], ["a"])
    assert values == pytest.approx([3, 1, 1 / 3, 1 / 3, 1 / 3, 0, 0, 0.8, 0, 0])
    assert query_features("", [], [], []) == [0.0] * 10
    with pytest.raises(ValueError, match="finite"):
        query_features("x", [float("nan")], [1], ["a"])


def test_labels_keep_fixed_on_ties():
    # 모드 점수가 동률이면 balanced, 그다음 keyword-heavy를 우선한다.
    from fusion.adaptive import training_labels

    assert training_labels([[1, 1, 1], [1, 0, 1], [0, 0, 1]]).tolist() == [1, 0, 2]


@pytest.mark.parametrize("kind", ["logistic", "mlp", "tree"])
def test_router_roundtrip_and_train_guard(tmp_path, kind):
    # Train 외 학습을 거부하고, 모델 저장·복원 후에도 예측을 유지한다.
    from fusion.adaptive import Router

    x = np.zeros((60, 10))
    x[:, 0] = np.repeat([0, 10, 20], 20)
    quality = np.repeat(np.eye(3), 20, axis=0)
    with pytest.raises(ValueError, match="train"):
        Router.fit(x, quality, weights=[0.2, 0.05, 0], kind=kind, split="test")
    model = Router.fit(x, quality, weights=[0.2, 0.05, 0], kind=kind, split="train")
    assert model.select(x).tolist() == np.repeat([0, 1, 2], 20).tolist()
    path = tmp_path / "router.json"
    model.save(path)
    assert Router.load(path).select(x).tolist() == model.select(x).tolist()


def test_constant_router_keeps_fixed_results_and_encodes_once(sources):  # noqa: F811
    # Balanced 고정 시 Fixed와 결과가 같고 query를 한 번만 인코딩한다.
    from fusion.adaptive import AdaptiveRetriever, Router
    from fusion.score import ScoreFusionRetriever

    sparse, dense = sources
    router = Router.fit(
        np.zeros((2, 10)),
        np.ones((2, 3)),
        weights=[0.2, 0.05, 0],
        kind="logistic",
        split="train",
    )
    adaptive = AdaptiveRetriever(sparse, dense, router)
    hits, details = adaptive.search_details("고양이", 10)
    assert dense.embedder.calls == ["고양이"]
    assert hits == ScoreFusionRetriever(sparse, dense).search("고양이", 10)
    assert details["mode"] == "balanced"
    assert details["feature_ms"] >= 0 and details["router_ms"] >= 0
    assert adaptive.search("  ", 1) == []
    with pytest.raises(ValueError):
        adaptive.search("a", 0)


def test_features_use_stable_ids_for_tied_top_ten():
    # 점수 동률은 문서 ID로 정렬해 Top-10 overlap을 일관되게 계산한다.
    from fusion.adaptive import query_features

    ids = [f"{i:02}" for i in range(11)]
    assert query_features("x", [0] * 10 + [1], [1] * 11, ids)[-1] == 0
    assert query_features("x", [1] + [0] * 10, [1] * 11, ids)[-1] == 0.1


@pytest.mark.parametrize("kind", ["logistic", "mlp", "tree"])
def test_binary_classes_without_balanced_roundtrip(tmp_path, kind):
    # Balanced 없는 두 클래스 모델도 저장·복원 후 예측을 유지한다.
    from fusion.adaptive import Router

    x = np.zeros((40, 10))
    x[20:, 0] = 100
    q = np.array([[1, 0, 0]] * 20 + [[0, 0, 1]] * 20)
    model = Router.fit(x, q, weights=[0.2, 0.05, 0], kind=kind, split="train")
    model.save(tmp_path / "model.json")
    assert (
        Router.load(tmp_path / "model.json").select(x).tolist() == [0] * 20 + [2] * 20
    )


@pytest.mark.parametrize("label, score", [(0, 1.0), (2, 0.8)])
def test_nondefault_modes_change_fusion_weight(sources, label, score):  # noqa: F811
    # Keyword-heavy와 semantic-heavy가 각 weight로 검색 점수를 계산한다.
    from fusion.adaptive import AdaptiveRetriever, Router

    quality = np.zeros((1, 3))
    quality[0, label] = 1
    model = Router.fit(
        np.zeros((1, 10)), quality, weights=[0.2, 0.05, 0], kind="tree", split="train"
    )
    sparse, dense = sources
    results = AdaptiveRetriever(sparse, dense, model).search("고양이", 2)
    assert [r.document_id for r in results] == ["a", "z"]
    assert results[0].score == pytest.approx(score)
    assert results[1].score == pytest.approx(0.8)


def test_tree_preserves_sklearn_double_threshold_comparison():
    # Tree 경계에서 float32 입력과 float64 임계값의 비교를 유지한다.
    from fusion.adaptive import FEATURES, Router

    state = dict(
        version=1,
        features=list(FEATURES),
        weights=[0.2, 0.05, 0],
        kind="tree",
        classes=[1, 2],
        mean=[0.0] * 10,
        scale=[1.0] * 10,
        tree=dict(
            left=[1, -1, -1],
            right=[2, -1, -1],
            feature=[0, -2, -2],
            threshold=[-0.09659633412957191, -2, -2],
            value=[[0.5, 0.5], [1, 0], [0, 1]],
        ),
    )
    x = np.zeros((1, 10))
    # 입력은 임계값보다 크다. 임계값까지 float32로 반올림하면 분기가 달라진다.
    x[0, 0] = -0.09659633040428162
    assert Router(state).select(x).tolist() == [2]


def test_router_accepts_validation_selected_weights(tmp_path):
    from fusion.adaptive import Router

    model = Router.fit(
        np.zeros((1, 10)),
        np.ones((1, 3)),
        weights=[0.3, 0.1, 0.01],
        kind="tree",
        split="train",
    )
    model.save(tmp_path / "router.json")
    assert Router.load(tmp_path / "router.json").state["weights"] == [0.3, 0.1, 0.01]


@pytest.mark.parametrize("classes", [[0, 1, 2], [0, 2]])
def test_mlp_portable_predictions_match_sklearn(monkeypatch, classes):
    from sklearn.neural_network import MLPClassifier

    from fusion.adaptive import Router

    fitted = []
    original = MLPClassifier.fit

    def capture(self, x, y):
        result = original(self, x, y)
        fitted.append(self)
        return result

    monkeypatch.setattr(MLPClassifier, "fit", capture)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(120, 10))
    labels = np.tile(classes, 120 // len(classes))
    quality = np.eye(3)[labels]
    router = Router.fit(x, quality, weights=[0.2, 0.05, 0], kind="mlp", split="train")
    z = (x - router.state["mean"]) / router.state["scale"]
    assert router.select(x).tolist() == fitted[0].predict(z).tolist()

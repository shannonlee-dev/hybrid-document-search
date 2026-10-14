"""Check E2E timing boundaries without downloading a model."""

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("faiss")
pytest.importorskip("bm25s")
pytest.importorskip("sklearn")

from data.loader import Document  # noqa: E402
from retrievers.dense import DenseConfig  # noqa: E402
from scripts import benchmark_score_fusion_e2e as bench  # noqa: E402


def test_every_request_encodes_and_rotation_excludes_warmup(monkeypatch):
    clock = [0.0]
    events = []
    prepared = bench._PreparedQuery(DenseConfig())

    class Encoder:
        def encode_query(self, query):
            events.append(("encode", query))
            clock[0] += 0.002
            return np.array([[1, 0]], dtype=np.float32)

    class Method:
        def __init__(self, name):
            self.name = name

        def search(self, query, k):
            assert events[-1] == ("sync", None)
            assert k == 10
            prepared.encode_query(query)
            events.append((self.name, query))
            clock[0] += 0.003
            return [SimpleNamespace(document_id="doc")]

    def _sync():
        events.append(("sync", None))

    monkeypatch.setattr(bench, "perf_counter", lambda: clock[0])
    methods = {name: Method(name) for name in ("dense", "rrf", "score_0.05")}
    samples, runs = bench._measure(
        methods,
        Encoder(),
        prepared,
        {"q1": "one", "q2": "two"},
        ["q1", "q2"],
        _sync,
        warmup=10,
        repeats=10,
    )
    # 방식별 준비 요청 10회 후 질의 2개를 각각 10회 측정한다.
    assert sum(e[0] == "encode" for e in events) == 3 * (10 + 20)
    calls = [e[0] for e in events if e[0] in methods][30:]
    assert calls[:6] == ["dense", "rrf", "score_0.05", "rrf", "score_0.05", "dense"]
    assert calls[6:9] == ["rrf", "score_0.05", "dense"]
    for name, rows in samples.items():
        assert len(rows) == 20
        assert {row["query_id"] for row in rows} == {"q1", "q2"}
        assert set(runs[name]) == {"q1", "q2"}
        for row in rows:
            assert row["ms"] == pytest.approx(5)
            assert row["embedding_ms"] == pytest.approx(2)
            assert row["retrieval_fusion_ms"] == pytest.approx(3)
        assert bench._summary([row["ms"] for row in rows])[
            "median_ms"
        ] == pytest.approx(5)


def test_prepared_vector_is_single_use_and_query_bound():
    prepared = bench._PreparedQuery(DenseConfig())
    with pytest.raises(ValueError):
        prepared.encode_query("q")
    vector = np.eye(2, dtype=np.float32)[:1]
    prepared._set("q", vector)
    with pytest.raises(ValueError):
        prepared.encode_query("other")
    assert prepared.encode_query("q") is vector
    with pytest.raises(ValueError):
        prepared.encode_query("q")


def test_real_retrievers_receive_fresh_vectors():
    prepared = bench._PreparedQuery(DenseConfig())
    docs = [Document("a", "고양이 동물"), Document("b", "우주 행성")]
    vectors = np.eye(2, dtype=np.float32)
    methods = bench._build_methods(docs, vectors, prepared)

    class Encoder:
        calls = 0

        def encode_query(self, query):
            self.calls += 1
            return vectors[[0 if query == "고양이" else 1]]

    encoder = Encoder()
    samples, runs = bench._measure(
        methods,
        encoder,
        prepared,
        {"q1": "고양이", "q2": "우주"},
        ["q1", "q2"],
        lambda: None,
        warmup=1,
        repeats=2,
    )
    assert encoder.calls == 15
    for name in methods:
        assert len(samples[name]) == 4
        assert runs[name]["q1"][0].document_id == "a"
        assert runs[name]["q2"][0].document_id == "b"


def test_summary_uses_pooled_samples_and_linear_p95():
    assert bench._summary([1, 2, 3, 10]) == {
        "mean_ms": 4,
        "median_ms": 2.5,
        "p95_ms": pytest.approx(8.95),
        "sample_count": 4,
    }


def test_existing_output_rejected_before_loading(tmp_path):
    with pytest.raises(FileExistsError):
        bench.main(["--output", str(tmp_path)])


def test_changed_rankings_fail_instead_of_publishing_samples():
    prepared = bench._PreparedQuery(DenseConfig())

    class Encoder:
        def encode_query(self, query):
            return np.eye(2, dtype=np.float32)[:1]

    class Unstable:
        calls = 0

        def search(self, query, k):
            prepared.encode_query(query)
            self.calls += 1
            return [SimpleNamespace(document_id=str(self.calls))]

    with pytest.raises(ValueError, match="unstable ranking"):
        bench._measure(
            {"dense": Unstable()},
            Encoder(),
            prepared,
            {"q": "query"},
            ["q"],
            lambda: None,
            warmup=1,
            repeats=2,
        )


@pytest.mark.parametrize("option", ["--warmup", "--repeats", "--threads"])
def test_nonpositive_counts_rejected(tmp_path, option):
    with pytest.raises(SystemExit):
        bench.main(["--output", str(tmp_path / "new"), option, "0"])
    assert not (tmp_path / "new").exists()


def test_pinned_cache_failure_happens_before_model_setup(tmp_path, monkeypatch):
    def _reject(*args):
        raise ValueError("data.json checksum differs from expected summary")

    monkeypatch.setattr(bench, "_load_cache", _reject)
    monkeypatch.setattr(
        bench,
        "configure_runtime",
        lambda *a, **kw: pytest.fail("runtime setup before provenance check"),
    )
    with pytest.raises(ValueError, match="checksum"):
        bench.main(["--output", str(tmp_path / "new")])
    assert not (tmp_path / "new").exists()


def test_method_label_and_actual_weight_follow_same_setting(monkeypatch):
    monkeypatch.setattr(bench, "DEFAULT_BM25_WEIGHT", 0.1)
    prepared = bench._PreparedQuery(DenseConfig())
    methods = bench._build_methods(
        [Document("a", "고양이"), Document("b", "우주")],
        np.eye(2, dtype=np.float32),
        prepared,
    )
    assert "score_0.1" in methods
    prepared._set("고양이", np.array([[1, 0]], dtype=np.float32))
    assert methods["score_0.1"].search("고양이", 1)[0].score == pytest.approx(1.1)

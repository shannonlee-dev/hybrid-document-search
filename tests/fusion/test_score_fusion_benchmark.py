"""Exercise offline evaluation with real BM25/FAISS and fixed tiny embeddings."""

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("faiss")
pytest.importorskip("bm25s")
pytest.importorskip("sklearn")

from scripts.benchmark_score_fusion import main  # noqa: E402


@pytest.mark.parametrize("corrected", [False, True])
def test_benchmark_reports_scores_splits_and_refuses_overwrite(tmp_path, corrected):
    cache = tmp_path / "cache"
    cache.mkdir()
    data = {
        "name": "fixture",
        "documents": [
            {"document_id": "a", "text": "고양이 동물"},
            {"document_id": "b", "text": "강아지 동물"},
            {"document_id": "c", "text": "우주 행성"},
        ],
        "query_ids": ["q1", "q2"],
        "queries": {"q1": "고양이", "q2": "우주"},
        "qrels": {"q1": {"a": 1}, "q2": {"c": 1}},
        "splits": {"train": ["q1"], "dev": ["q2"]},
    }
    (cache / "data.json").write_text(json.dumps(data), encoding="utf-8")
    np.save(cache / "document-vectors.npy", np.eye(3, dtype=np.float32))
    np.save(cache / "query-vectors.npy", np.eye(3, dtype=np.float32)[[0, 2]])
    output = tmp_path / "result"
    arguments = [
        "--cache-dir",
        str(cache),
        "--output",
        str(output),
        "--split",
        "train",
        "--weights",
        "0",
        "0.05",
        "--timing-queries",
        "1",
        "--repeats",
        "1",
    ]
    if corrected:
        arguments.append("--correction")
    assert main(arguments) == 0
    report = json.loads((output / "summary.json").read_text())
    assert report["query_count"] == 1
    assert report["corpus_size"] == 3
    assert report["quality"]["score_0.05"]["ndcg@10"] == 1
    assert report["quality"]["dense"]["recall@10"] == 1
    assert report["latency"]["score_0.05"]["sample_count"] == 1
    assert report["latency_excludes"] == ["model_inference", "initialization", "HTTP"]
    runs = json.loads((output / "runs.json").read_text())
    assert set(runs["score_0.05"]) == {"q1"}
    assert runs["score_0.05"]["q1"][0]["document_id"] == "a"
    assert runs["score_0.05"]["q1"][0]["score"] == pytest.approx(1.05)
    from evaluation.latency import latency_summary
    from evaluation.metrics import evaluate_query, evaluate_run
    from retrievers.base import SearchResult

    per_query = json.loads((output / "per-query.json").read_text())
    timings = json.loads((output / "timings.json").read_text())
    for name, run in runs.items():
        hits = {qid: [SearchResult(**hit) for hit in rows] for qid, rows in run.items()}
        assert report["quality"][name] == evaluate_run(hits, {"q1": {"a": 1}})
        assert per_query[name]["q1"] == evaluate_query(hits["q1"], {"a": 1})
    for name, samples in timings.items():
        assert report["latency"][name] == latency_summary([s["ms"] for s in samples])
    assert ("corrected_score" in runs) == corrected
    original = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        main(arguments)
    assert (output / "summary.json").read_bytes() == original


def test_pca_ablation_does_not_mutate_inputs():
    from scripts.benchmark_score_fusion import _PcaCorrection

    documents = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    queries = np.array([[0, -1]], dtype=np.float32)
    original = documents.copy()
    correction = _PcaCorrection(documents)
    doc, query = correction._transform(documents), correction._transform(queries)
    np.testing.assert_array_equal(documents, original)
    np.testing.assert_allclose(np.linalg.norm(doc, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(query, axis=1), 1, atol=1e-6)
    # 2차원에서는 모든 방향을 같은 비율로 축소하므로 정규화 후 원래 방향이 유지됩니다.
    np.testing.assert_allclose(query, [[0, -1]], atol=1e-6)


@pytest.mark.parametrize(
    ("documents", "vectors"),
    [
        pytest.param([[1, 0], [1, 0]], [[1, 0], [1, 0]], id="identical-documents"),
        pytest.param([[1, 0], [0, 1]], [[1, 0], [0.5, 0.5]], id="query-at-mean"),
        pytest.param([[1, 0], [0, 1]], [[1e20, 0]], id="overflowing-norm"),
    ],
)
def test_pca_ablation_rejects_invalid_corrected_norms(documents, vectors):
    from scripts.benchmark_score_fusion import _PcaCorrection

    correction = _PcaCorrection(np.array(documents, dtype=np.float32))
    with np.errstate(over="ignore", divide="raise", invalid="raise"):
        with pytest.raises(ValueError):
            correction._transform(np.array(vectors, dtype=np.float32))


@pytest.fixture
def cache(tmp_path):
    directory = tmp_path / "cache"
    directory.mkdir()
    data = {
        "name": "synthetic",
        "documents": [
            {"document_id": "a", "text": "고양이"},
            {"document_id": "b", "text": "우주"},
        ],
        "query_ids": ["q1", "q2"],
        "queries": {"q1": "고양이", "q2": "우주"},
        "qrels": {"q1": {"a": 1}, "q2": {"b": 1}},
        "splits": {"train": ["q1"], "dev": ["q2"]},
    }
    (directory / "data.json").write_text(json.dumps(data), encoding="utf-8")
    np.save(directory / "document-vectors.npy", np.eye(2, dtype=np.float32))
    np.save(directory / "query-vectors.npy", np.eye(2, dtype=np.float32))
    return directory


def test_unrecorded_model_is_not_reported_as_bge_m3(cache, tmp_path):
    output = tmp_path / "output"
    main(["--cache-dir", str(cache), "--output", str(output), "--repeats", "1"])
    report = json.loads((output / "summary.json").read_text())
    assert report["model_config"] is None
    assert report["provenance_status"] == "unverified"
    assert report["weights"] == [0.05]
    assert (
        report["latency_scope"] == "CPU retrieval and fusion from cached query vectors"
    )


def test_overlapping_train_dev_is_rejected(cache):
    from scripts.benchmark_score_fusion import _load_cache

    path = cache / "data.json"
    data = json.loads(path.read_text())
    data["splits"]["dev"] = ["q1"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="overlap"):
        _load_cache(cache, "train")


def test_expected_summary_detects_changed_mapping(cache, tmp_path):
    baseline = tmp_path / "baseline"
    main(["--cache-dir", str(cache), "--output", str(baseline), "--repeats", "1"])
    reference = baseline / "summary.json"
    args = [
        "--cache-dir",
        str(cache),
        "--output",
        str(tmp_path / "output"),
        "--expected-summary",
        str(reference),
        "--repeats",
        "1",
    ]
    path = cache / "data.json"
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="checksum"):
        main(args)
    assert not (tmp_path / "output").exists()
    path.write_bytes(original)
    assert main(args) == 0


def test_close_alpha_values_cannot_overwrite_one_another(cache, tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "--cache-dir",
                str(cache),
                "--output",
                str(tmp_path / "output"),
                "--weights",
                "0.050000001",
                "0.050000002",
            ]
        )
    assert not (tmp_path / "output").exists()


def test_reproduction_rejects_changed_quality_and_rankings(cache, tmp_path):
    output = tmp_path / "baseline"
    common = ["--cache-dir", str(cache), "--repeats", "1"]
    main([*common, "--output", str(output)])
    reference = output / "summary.json"
    report = json.loads(reference.read_text())
    report["quality"]["score_0.05"]["ndcg@10"] = -123
    reference.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="quality"):
        main(
            [
                *common,
                "--output",
                str(tmp_path / "bad"),
                "--expected-summary",
                str(reference),
            ]
        )
    assert not (tmp_path / "bad").exists()

    rankings = json.loads((output / "runs.json").read_text())
    rankings["dense"]["q1"].reverse()
    expected_runs = tmp_path / "changed-runs.json"
    expected_runs.write_text(json.dumps(rankings))
    with pytest.raises(ValueError, match="ranking"):
        main(
            [
                *common,
                "--output",
                str(tmp_path / "bad-ranks"),
                "--expected-runs",
                str(expected_runs),
            ]
        )
    assert not (tmp_path / "bad-ranks").exists()


def test_reports_hashes_and_sweep_uses_one_snapshot(cache, tmp_path, monkeypatch):
    from indexing.faiss_index import FaissIndex
    from scripts.benchmark_score_fusion import _sha256

    snapshots = []
    original = FaissIndex.snapshot_vectors

    def _snapshot(index):
        vectors = original(index)
        snapshots.append(vectors)
        return vectors

    monkeypatch.setattr(FaissIndex, "snapshot_vectors", _snapshot)
    output = tmp_path / "output"
    main(
        [
            "--cache-dir",
            str(cache),
            "--output",
            str(output),
            "--weights",
            "0",
            "0.05",
            "0.1",
            "--repeats",
            "1",
        ]
    )
    assert len(snapshots) == 1
    report = json.loads((output / "summary.json").read_text())
    for name in ("runs.json", "per-query.json", "timings.json"):
        assert report["artifact_sha256"][name] == _sha256(output / name)
    for name in ("fusion/hybrid.py", "indexing/vectors.py"):
        from pathlib import Path

        assert report["code_sha256"][name] == _sha256(Path(name))
    # 재실행에서도 기준 검색 품질과 순위가 같으면 검증을 통과해야 한다.
    main(
        [
            "--cache-dir",
            str(cache),
            "--output",
            str(tmp_path / "again"),
            "--weights",
            "0",
            "0.05",
            "0.1",
            "--repeats",
            "1",
            "--expected-summary",
            str(output / "summary.json"),
            "--expected-runs",
            str(output / "runs.json"),
        ]
    )

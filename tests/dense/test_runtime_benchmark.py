"""Dense 복원, revision·입력 무결성, 실패 기록과 기존 파일 보존을 검증한다."""

import hashlib
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("faiss")

CANDIDATES = (
    ("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a"),
    ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"),
    ("nlpai-lab/KURE-v1", "8b418a58414668e75532ed045c22d9ca018ae2b2"),
)


@pytest.fixture
def inputs(tmp_path, dense_corpus_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes(dense_corpus_path.read_bytes())
    queries = tmp_path / "queries_train.jsonl"
    queries.write_text('{"query_id": "q1", "text": "고양이"}\n')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_revision": "fixture-only",
                "output_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (corpus, queries)
                },
            }
        )
    )
    from evaluation.dense_runtime import DenseRuntimeConfig

    return DenseRuntimeConfig(
        corpus=corpus,
        queries=queries,
        manifest=manifest,
        expected_documents=3,
        device="cpu",
        batch_size=1,
        top_k=2,
        warmup=1,
        repeats=2,
        index_root=tmp_path / "indexes",
    )


@pytest.mark.parametrize("top_k", [2, 10])
def test_build_and_reload_without_document_embedding(
    inputs, model_stub, monkeypatch, top_k
):
    from evaluation import benchmark, latency
    from evaluation.dense_runtime import build, validate
    from retrievers.dense import DenseEmbedder

    inputs.top_k = top_k
    index = inputs.index_root / "fixture"
    before = build(inputs, "nlpai-lab/KURE-v1", index)
    baseline = json.loads((index / "benchmark_build.json").read_text())

    def _forbidden(*args, **kwargs):
        pytest.fail("validation must not re-embed documents or calculate quality")

    monkeypatch.setattr(DenseEmbedder, "encode_documents", _forbidden)
    monkeypatch.setattr(benchmark, "evaluate_query", _forbidden)
    monkeypatch.setattr(benchmark, "evaluate_run", _forbidden)
    after = validate(inputs, index)
    assert before["document_count"] == inputs.expected_documents
    assert before["embedding_seconds"] >= 0
    assert before["faiss_build_seconds"] >= 0
    assert before["build_seconds"] >= (
        before["embedding_seconds"] + before["faiss_build_seconds"]
    )
    assert before["total_preparation_seconds"] >= before["build_seconds"]
    smoke = after["smoke_validation"]
    assert len(smoke["hits"]) == min(top_k, inputs.expected_documents)
    assert smoke["hits"] == baseline["hits"]
    assert smoke["full_mapping_matches"] is True
    assert smoke["document_embeddings_recreated"] is False
    raw = after["latency_samples_ms"]
    assert len(raw) == 1 and len(raw[0]) == inputs.repeats
    summary = latency.latency_summary(raw[0])
    assert after["latency_samples"] == inputs.repeats
    assert after["latency_mean_seconds"] == pytest.approx(summary["mean_ms"] / 1000)
    assert after["latency_p95_seconds"] == pytest.approx(summary["p95_ms"] / 1000)


@pytest.mark.parametrize("change", ["corpus", "queries", "count"])
def test_rejects_changed_inputs_before_embedding(inputs, change, monkeypatch):
    from evaluation.dense_runtime import build
    from retrievers.dense import DenseEmbedder

    def _forbidden(*args, **kwargs):
        pytest.fail("invalid inputs must be rejected before embedding")

    monkeypatch.setattr(DenseEmbedder, "encode_documents", _forbidden)

    if change != "count":
        path = getattr(inputs, change)
        path.write_text(path.read_text() + "\n")
    else:
        inputs.expected_documents = 10000
    with pytest.raises(ValueError, match="hash|count"):
        build(inputs, "nlpai-lab/KURE-v1", inputs.index_root / "fixture")


def test_rejects_saved_mapping_that_differs_from_corpus(inputs, model_stub):
    from evaluation.dense_runtime import build, validate
    from retrievers.dense import DenseRetriever

    index = inputs.index_root / "fixture"
    build(inputs, "intfloat/multilingual-e5-base", index)
    loaded = DenseRetriever.load(index)
    loaded.documents = tuple(reversed(loaded.documents))
    loaded.save(index)
    with pytest.raises(ValueError, match="mapping"):
        validate(inputs, index)


@pytest.mark.parametrize("failed_stage", ["build", "validate"])
def test_failure_is_persisted_without_fabricated_measurements(
    inputs, monkeypatch, tmp_path, failed_stage
):
    from scripts import benchmark_dense_runtime

    def _fail(args, model, index, phase):
        if phase == failed_stage:
            raise RuntimeError("CUDA out of memory")
        return {
            "embedding_seconds": 1.5,
            "faiss_build_seconds": 0.01,
            "total_preparation_seconds": 2.0,
        }

    monkeypatch.setattr(benchmark_dense_runtime, "_run_worker", _fail)
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit) as exc:
        benchmark_dense_runtime.main(_cli_args(inputs, output))
    assert exc.value.code == 2
    report = json.loads(output.read_text())
    assert len(report["models"]) == 3
    for row in report["models"]:
        assert row["status"] == "failed"
        assert "CUDA out of memory" in row["error"]
        assert row["failed_stage"] == failed_stage
        for key, expected in (
            ("embedding_seconds", 1.5),
            ("faiss_build_seconds", 0.01),
            ("total_preparation_seconds", 2.0),
        ):
            assert row[key] == (None if failed_stage == "build" else expected)
        assert row["latency_mean_seconds"] is None
        assert row["latency_p95_seconds"] is None
        assert row["latency_samples"] is None


def test_existing_report_is_preserved_before_any_work(tmp_path, monkeypatch):
    from scripts import benchmark_dense_runtime

    output = tmp_path / "report.json"
    original = '{"models": [{"status": "completed"}]}\n'
    output.write_text(original)

    def _unexpected(*args):
        pytest.fail("existing output must be rejected before input loading or workers")

    monkeypatch.setattr(
        benchmark_dense_runtime.dense_runtime, "check_inputs", _unexpected
    )
    monkeypatch.setattr(benchmark_dense_runtime, "_run_worker", _unexpected)
    with pytest.raises(SystemExit) as exc:
        benchmark_dense_runtime.main(["--output", str(output)])
    assert exc.value.code == 2
    assert output.read_text() == original


@pytest.mark.parametrize("case", ["existing_last", "output_inside"])
def test_rejects_unsafe_index_paths_before_execution(
    inputs, tmp_path, monkeypatch, capsys, case
):
    from scripts import benchmark_dense_runtime as cli

    def _unexpected_worker(*args):
        pytest.fail("all index paths must be validated before any worker runs")

    monkeypatch.setattr(cli, "_run_worker", _unexpected_worker)
    first_index = inputs.index_root / "foo--bar"
    last_index = inputs.index_root / "other--model"
    output = tmp_path / "report.json"
    if case == "existing_last":
        last_index.mkdir(parents=True)
        marker = last_index / "existing.txt"
        marker.write_text("preserve me")
        error = "index already exists"
    else:
        output = last_index / "report.json"
        error = "--output must not be inside an index directory"
    with pytest.raises(SystemExit) as exc:
        cli.main(_cli_args(inputs, output, "foo/bar", "other/model"))
    assert exc.value.code == 2
    assert error in capsys.readouterr().err
    assert not output.exists()
    if case == "existing_last":
        assert marker.read_text() == "preserve me"
        assert not first_index.exists()
    else:
        assert not inputs.index_root.exists()


@pytest.mark.parametrize("model,revision", CANDIDATES)
@pytest.mark.parametrize("saved_revision", [None, "f" * 40, "missing"])
def test_validation_rejects_changed_metadata_revision_before_model_load(
    inputs, model_stub, monkeypatch, model, revision, saved_revision
):
    from evaluation.dense_runtime import build, validate

    index = inputs.index_root / "fixture"
    build(inputs, model, index)
    path = index / "metadata.json"
    metadata = json.loads(path.read_text())
    if saved_revision == "missing":
        metadata["embedding_config"].pop("revision", None)
    else:
        metadata["embedding_config"]["revision"] = saved_revision
    path.write_text(json.dumps(metadata))

    def _forbidden(*args, **kwargs):
        pytest.fail("revision mismatch must fail before loading the model")

    monkeypatch.setattr(model_stub, "__init__", _forbidden)
    with pytest.raises(ValueError, match="revision"):
        validate(inputs, index)


@pytest.mark.parametrize("model,revision", CANDIDATES)
@pytest.mark.parametrize("phase", ["build", "validate"])
@pytest.mark.parametrize("actual", [None, "f" * 40])
def test_rejects_unverifiable_or_different_loaded_revision(
    inputs, model_stub, monkeypatch, model, revision, phase, actual
):
    from evaluation.dense_runtime import build, validate

    index = inputs.index_root / "fixture"
    if phase == "validate":
        build(inputs, model, index)
    monkeypatch.setattr(
        model_stub,
        "_first_module",
        lambda self: SimpleNamespace(
            auto_model=SimpleNamespace(config=SimpleNamespace(_commit_hash=actual))
        ),
    )
    with pytest.raises(ValueError, match="loaded model revision"):
        if phase == "build":
            build(inputs, model, index)
        else:
            validate(inputs, index)


def _cli_args(inputs, output, *models):
    options = {
        "corpus": inputs.corpus,
        "queries": inputs.queries,
        "manifest": inputs.manifest,
        "expected-documents": inputs.expected_documents,
        "index-root": inputs.index_root,
        "output": output,
        "device": inputs.device,
        "top-k": inputs.top_k,
        "warmup": inputs.warmup,
        "repeats": inputs.repeats,
    }
    args = [item for key, value in options.items() for item in ("--" + key, str(value))]
    for model in models:
        args.extend(["--model", str(model)])
    return args

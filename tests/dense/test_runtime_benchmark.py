"""Verify Dense restoration, input/revision integrity and failure recording."""

import hashlib
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("faiss")

_CANDIDATES = (
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
    assert before["embedding_seconds"] >= 0
    assert before["faiss_build_seconds"] >= 0
    assert before["build_seconds"] >= (
        before["embedding_seconds"] + before["faiss_build_seconds"]
    )
    assert before["total_preparation_seconds"] >= before["build_seconds"]
    smoke = after["smoke_validation"]
    assert smoke["hits"] == baseline["hits"]
    assert smoke["full_mapping_matches"] is True
    assert smoke["document_embeddings_recreated"] is False
    raw = after["latency_samples_ms"]
    assert len(raw) == 1 and len(raw[0]) == inputs.repeats
    summary = latency.latency_summary(raw[0])
    assert after["latency_samples"] == inputs.repeats
    assert after["latency_mean_seconds"] == pytest.approx(summary["mean_ms"] / 1000)
    assert after["latency_p95_seconds"] == pytest.approx(summary["p95_ms"] / 1000)


@pytest.mark.parametrize("change", ["corpus", "queries"])
def test_rejects_changed_inputs_before_embedding(inputs, change, monkeypatch):
    from evaluation.dense_runtime import build
    from retrievers.dense import DenseEmbedder

    def _forbidden(*args, **kwargs):
        pytest.fail("invalid inputs must be rejected before embedding")

    monkeypatch.setattr(DenseEmbedder, "encode_documents", _forbidden)

    path = getattr(inputs, change)
    path.write_text(path.read_text() + "\n")
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


@pytest.mark.parametrize("model,revision", _CANDIDATES)
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


@pytest.mark.parametrize("model,revision", _CANDIDATES)
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

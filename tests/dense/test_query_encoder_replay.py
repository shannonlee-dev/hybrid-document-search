"""Check frozen selection, per-query metrics, and replay input integrity."""

import hashlib
import json
from math import log2

import pytest

np = pytest.importorskip("numpy")

from evaluation import query_encoder as encoder  # noqa: E402


@pytest.fixture
def replay_inputs(tmp_path):
    data, cache, scores, adapted = [
        tmp_path / name for name in ("data", "cache", "scores", "adapted")
    ]
    for directory in (data, cache, scores, adapted):
        directory.mkdir()

    def save(path, value):
        path.write_text(json.dumps(value) + "\n")

    def rows(path, value):
        path.write_text("".join(json.dumps(row) + "\n" for row in value))

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    documents = [
        {"document_id": "a", "text": "card"},
        {"document_id": "b", "text": "lost"},
    ]
    rows(data / "corpus.jsonl", documents)
    save(scores / "corpus.json", documents)
    cached_queries, splits = [], {}
    for split in ("train", "validation", "test"):
        queries = [
            dict(
                query_id=f"{split}-{i}",
                text=f"{split} {i}",
                provenance="base",
                parent_doc_id=f"{split}-{i}",
            )
            for i in range(2)
        ]
        rows(data / f"queries_{split}.jsonl", queries)
        rows(
            data / f"qrels_{split}.jsonl",
            [
                dict(query_id=q["query_id"], document_id=d["document_id"], relevance=3)
                for q, d in zip(queries, documents, strict=True)
            ],
        )
        splits[split] = {"query_ids": [q["query_id"] for q in queries]}
        if split != "train":
            cached_queries.extend([{**q, "split": split} for q in queries])
            vectors = np.eye(2) if split == "validation" else np.eye(2)[::-1]
            np.save(adapted / f"selected-{split}-vectors.npy", vectors)
    save(scores / "queries.json", cached_queries)
    np.save(scores / "dense.npy", np.array([[1.0, 0.0]] * 4))
    np.save(scores / "sparse.npy", np.zeros((4, 2)))
    np.save(cache / "document-vectors.npy", np.eye(2))
    manifest = {
        "schema_version": "1",
        "dataset_revision": "1" * 40,
        "splits": ["train", "validation", "test"],
        "primary_slice_query_ids": ["test-0", "test-1"],
        "output_sha256": {p.name: sha(p) for p in data.glob("*.jsonl")},
        "output_counts": {
            p.name: len(p.read_text().splitlines()) for p in data.glob("*.jsonl")
        },
    }
    save(data / "manifest.json", manifest)
    save(
        cache / "encoding.json",
        {
            "model": {"model_name": "fixture", "revision": "2" * 40},
            "dataset_manifest_sha256": sha(data / "manifest.json"),
            "document_ids_sha256": hashlib.sha256(b"a\nb").hexdigest(),
            "vector_sha256": {
                "document-vectors.npy": sha(cache / "document-vectors.npy")
            },
            "splits": splits,
        },
    )
    (adapted / "selected-last-block.pt").write_bytes(b"checkpoint")
    save(
        adapted / "frozen.json",
        {
            "checkpoint_sha256": sha(adapted / "selected-last-block.pt"),
            "selected_step": 64,
        },
    )
    return data, cache, scores, {"seed43": adapted}, tmp_path / "output"


def test_replay_exports_actual_test_ranks_without_reselecting_encoder(replay_inputs):
    frozen = encoder.select_cached_encoders(*replay_inputs)
    assert frozen["selected_encoder"] == "seed43"
    report = encoder.evaluate_cached_encoders(*replay_inputs)
    # Test favors baseline, but test labels must never change validation selection.
    assert report["selected_encoder"] == "seed43"
    assert (
        report["models"]["baseline"]["test"]["ndcg@10"]
        > report["models"]["seed43"]["test"]["ndcg@10"]
    )
    assert report["models"]["seed43"]["test"]["mrr@10"] == 0.5
    output = replay_inputs[-1]
    records = [
        json.loads(line)
        for line in (output / "per-query.jsonl").read_text().splitlines()
    ]
    finance = [r for r in records if r["method"] == "seed43 + Fixed"]
    assert len(finance) == 2
    assert finance[0]["query_id"] == "test-0"
    assert finance[0]["top10_document_ids"] == ["b", "a"]
    assert finance[0]["relevant_rank_top10"] == 2
    assert finance[0]["p@1"] == 0
    assert finance[0]["recall@10"] == 1
    assert finance[0]["mrr@10"] == 0.5
    assert finance[0]["ndcg@10"] == pytest.approx(1 / log2(3))
    assert (
        finance[0]["selected_alpha_dense"] == frozen["models"]["seed43"]["alpha_dense"]
    )
    assert "text" not in finance[0]


@pytest.fixture
def bf16_baseline(replay_inputs, tmp_path):
    data, cache, _, _, _ = replay_inputs
    directory = tmp_path / "bf16-baseline"
    directory.mkdir()
    encoding = json.loads((cache / "encoding.json").read_text())
    metadata = {
        "dataset_manifest_sha256": hashlib.sha256(
            (data / "manifest.json").read_bytes()
        ).hexdigest(),
        "document_vectors_sha256": encoding["vector_sha256"]["document-vectors.npy"],
        "model": encoding["model"],
        "query_encoding": {
            "forward_dtype": "bfloat16",
            "storage_dtype": "float32",
            "max_tokens": 512,
            "batch_size": 16,
        },
        "splits": {},
    }
    for split in ("validation", "test"):
        path = directory / f"baseline-{split}-vectors.npy"
        np.save(path, np.eye(2, dtype=np.float32))
        metadata["splits"][split] = {
            "query_ids": encoding["splits"][split]["query_ids"],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (directory / "baseline-vectors.json").write_text(json.dumps(metadata))
    adapted = replay_inputs[3]["seed43"]
    for split in ("validation", "test"):
        path = adapted / f"selected-{split}-vectors.npy"
        np.save(path, np.load(path).astype(np.float32))
    (adapted / "protocol.json").write_text(
        json.dumps(
            {
                "precision": "FP32 parameters, BF16 autocast",
                "max_eval_tokens": 512,
                "revision": encoding["model"]["revision"],
            }
        )
    )
    return directory


def test_bf16_baseline_replaces_fp32_scores_and_keeps_both_alpha_policies(
    replay_inputs, bf16_baseline, tmp_path
):
    from evaluation.query_encoder_report import compact_export, verify_results

    frozen = encoder.select_cached_encoders(
        *replay_inputs,
        baseline_vectors_dir=bf16_baseline,
        shared_alpha=0.84,
    )
    assert frozen["models"]["baseline"]["validation_ndcg"] == 1
    assert frozen["models"]["baseline"]["alpha_dense"] == 0.01
    report = encoder.evaluate_cached_encoders(
        *replay_inputs, baseline_vectors_dir=bf16_baseline
    )
    assert report["models"]["baseline"]["test"]["ndcg@10"] == 1
    comparisons = report["comparisons"]
    assert comparisons["baseline + Fixed alpha=0.84"]["ndcg@10"] == 1
    assert comparisons["seed43 + Fixed alpha=0.84"]["mrr@10"] == 0.5
    assert comparisons["seed43 + Fixed alpha=0.84"]["ndcg@10"] == pytest.approx(
        1 / log2(3)
    )
    records = [
        json.loads(line)
        for line in (replay_inputs[-1] / "per-query.jsonl").read_text().splitlines()
    ]
    assert {r["selection_policy"] for r in records} == {
        "shared_alpha",
        "validation_selected",
    }
    target = tmp_path / "matched-compact"
    compact_export(replay_inputs[0], replay_inputs[-1], target)
    assert verify_results(replay_inputs[0], target)["rows"] == 8


@pytest.mark.parametrize(
    "tamper",
    [
        "query_order",
        "vector_bytes",
        "forward_dtype",
        "document_vectors",
        "adapted_precision",
    ],
)
def test_matched_baseline_refuses_wrong_identity_or_precision(
    replay_inputs, bf16_baseline, tamper
):
    metadata_path = bf16_baseline / "baseline-vectors.json"
    metadata = json.loads(metadata_path.read_text())
    if tamper == "query_order":
        metadata["splits"]["validation"]["query_ids"].reverse()
    elif tamper == "vector_bytes":
        np.save(
            bf16_baseline / "baseline-validation-vectors.npy",
            np.ones((2, 2), dtype=np.float32),
        )
    elif tamper == "forward_dtype":
        metadata["query_encoding"]["forward_dtype"] = "float32"
    elif tamper == "document_vectors":
        metadata["document_vectors_sha256"] = "0" * 64
    else:
        path = replay_inputs[3]["seed43"] / "protocol.json"
        protocol = json.loads(path.read_text())
        protocol["precision"] = "FP32 parameters and forward"
        path.write_text(json.dumps(protocol))
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        encoder.select_cached_encoders(
            *replay_inputs, baseline_vectors_dir=bf16_baseline, shared_alpha=0.84
        )


def test_matched_baseline_cannot_change_after_validation(replay_inputs, bf16_baseline):
    encoder.select_cached_encoders(
        *replay_inputs, baseline_vectors_dir=bf16_baseline, shared_alpha=0.84
    )
    np.save(
        bf16_baseline / "baseline-validation-vectors.npy",
        np.ones((2, 2), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="changed"):
        encoder.evaluate_cached_encoders(
            *replay_inputs, baseline_vectors_dir=bf16_baseline
        )


def test_prepare_scores_needs_no_other_experiment_cache(replay_inputs):
    pytest.importorskip("bm25s")
    data, cache, _, _, output = replay_inputs
    for split in ("validation", "test"):
        np.save(cache / f"query-vectors-{split}.npy", np.eye(2))
    encoding_path = cache / "encoding.json"
    encoding = json.loads(encoding_path.read_text())
    for split in ("validation", "test"):
        name = f"query-vectors-{split}.npy"
        encoding["vector_sha256"][name] = hashlib.sha256(
            (cache / name).read_bytes()
        ).hexdigest()
    encoding_path.write_text(json.dumps(encoding))
    encoder.prepare_scores(data, cache, output)
    assert np.load(output / "dense.npy").tolist() == [[1, 0], [0, 1]] * 2
    assert np.load(output / "sparse.npy").shape == (4, 2)
    assert (output / "protocol.json").is_file()


def test_verify_bundle_without_original_cache_or_checkpoint(replay_inputs, tmp_path):
    import shutil

    from evaluation.query_encoder_report import verify_results

    encoder.select_cached_encoders(*replay_inputs)
    encoder.evaluate_cached_encoders(*replay_inputs)
    moved = tmp_path / "moved"
    shutil.copytree(replay_inputs[-1], moved)
    for directory in replay_inputs[1:3]:
        shutil.rmtree(directory)
    shutil.rmtree(replay_inputs[3]["seed43"])
    result = verify_results(replay_inputs[0], moved)
    assert result["verified"] is True
    assert result["queries"] == 2


@pytest.mark.parametrize(
    "change", ["metric", "alpha", "missing_row", "duplicate_row", "document", "run_id"]
)
def test_verify_rejects_forged_results_even_if_file_hash_is_updated(
    replay_inputs, change
):
    from evaluation.query_encoder_report import verify_results

    encoder.select_cached_encoders(*replay_inputs)
    encoder.evaluate_cached_encoders(*replay_inputs)
    output = replay_inputs[-1]
    path = output / "per-query.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "metric":
        records[0]["p@1"] = 0.25
    elif change == "alpha":
        records[0]["selected_alpha_dense"] = 0.99
    elif change == "missing_row":
        records.pop()
    elif change == "duplicate_row":
        records.append(records[0])
    elif change == "document":
        records[0]["top10_document_ids"] = ["unknown"]
    else:
        records[0]["run_id"] = "another-run"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    manifest_path = output / "per-query-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_results(replay_inputs[0], output)


def test_evaluate_does_not_overwrite_completed_results(replay_inputs):
    encoder.select_cached_encoders(*replay_inputs)
    encoder.evaluate_cached_encoders(*replay_inputs)
    path = replay_inputs[-1] / "per-query.jsonl"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="exist"):
        encoder.evaluate_cached_encoders(*replay_inputs)
    assert path.read_bytes() == before


def test_compact_export_keeps_only_two_methods_and_no_arrays(replay_inputs, tmp_path):
    from evaluation.query_encoder_report import compact_export, verify_results

    encoder.select_cached_encoders(*replay_inputs)
    encoder.evaluate_cached_encoders(*replay_inputs)
    target = tmp_path / "compact"
    compact_export(replay_inputs[0], replay_inputs[-1], target)
    assert sorted(p.name for p in target.iterdir()) == [
        "frozen.json",
        "per-query-manifest.json",
        "per-query.jsonl",
        "protocol.json",
    ]
    assert verify_results(replay_inputs[0], target)["rows"] == 4
    assert (
        json.loads((target / "protocol.json").read_text())["training_data_audit"][
            "excluded_exact_duplicate_examples"
        ]
        == 0
    )


def test_compact_export_checks_canonical_manifest_identity(replay_inputs, tmp_path):
    from evaluation.query_encoder_report import compact_export

    encoder.select_cached_encoders(*replay_inputs)
    encoder.evaluate_cached_encoders(*replay_inputs)
    data = replay_inputs[0]
    prepared = json.loads((data / "manifest.json").read_text())
    canonical = tmp_path / "canonical.json"
    canonical.write_text(
        json.dumps(
            {
                "dataset": {
                    "manifest_sha256": hashlib.sha256(
                        (data / "manifest.json").read_bytes()
                    ).hexdigest(),
                    "input_sha256": prepared["output_sha256"],
                },
                "encoding": {"vector_sha256": {}},
                "code": {"git_commit": "0" * 40},
                "methods": {},
            }
        )
    )
    target = tmp_path / "compact"
    compact_export(data, replay_inputs[-1], target, canonical_summary=canonical)
    manifest = json.loads((target / "per-query-manifest.json").read_text())
    assert (
        manifest["canonical_reference"]["source_sha256"]
        == hashlib.sha256(canonical.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "filename", ["corpus.jsonl", "queries_validation.jsonl", "qrels_validation.jsonl"]
)
def test_replay_rejects_changed_data_even_when_row_ids_still_match(
    replay_inputs, filename
):
    data = replay_inputs[0]
    path = data / filename
    value = [json.loads(line) for line in path.read_text().splitlines()]
    if filename.startswith("qrels"):
        value[0]["relevance"] = 0
    else:
        value[0]["text"] += " changed"
    path.write_text("".join(json.dumps(row) + "\n" for row in value))
    with pytest.raises(ValueError, match="체크섬|checksum"):
        encoder.select_cached_encoders(*replay_inputs)


def test_replay_rejects_changed_document_vectors(replay_inputs):
    cache = replay_inputs[1]
    np.save(cache / "document-vectors.npy", np.eye(2)[::-1])
    with pytest.raises(ValueError, match="checksum"):
        encoder.select_cached_encoders(*replay_inputs)


def test_test_evaluation_rejects_validation_input_change(replay_inputs):
    encoder.select_cached_encoders(*replay_inputs)
    np.save(replay_inputs[2] / "sparse.npy", np.ones((4, 2)))
    with pytest.raises(ValueError, match="changed"):
        encoder.evaluate_cached_encoders(*replay_inputs)

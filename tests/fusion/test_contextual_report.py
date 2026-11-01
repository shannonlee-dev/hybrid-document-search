"""Protect DAT handoff metrics, routing metadata and artifact identity."""

import json

import pytest

from data.loader import Document
from evaluation.data import EvaluationDataset, Query

np = pytest.importorskip("numpy")


@pytest.fixture
def dataset():
    return EvaluationDataset(
        (Document("a", "dense body"), Document("b", "sparse body")),
        (Query("q1", "question"),),
        {"q1": {"b": 3}},
        "test",
        {"manifest_sha256": "a" * 64},
    )


def _routing_records(dataset, *, agreement=False):
    from evaluation.contextual_report import ranking_record

    top1 = ["b", "b"] if agreement else ["a", "b"]
    return [
        ranking_record(
            dataset,
            "q1",
            [1, 0],
            run_id="live",
            method="dat_fixed",
            alpha=0.84,
            top1=top1,
        ),
        ranking_record(
            dataset,
            "q1",
            [1, 0],
            run_id="live",
            method="dat",
            alpha=0.84 if agreement else 1 / 17,
            judged=not agreement,
            top1=top1,
            probabilities=None if agreement else [0.2, 0.8],
        ),
    ]


@pytest.mark.parametrize("agreement", [False, True])
def test_matched_routing_records_validate(dataset, agreement):
    from evaluation.contextual_report import validate_records

    checked = validate_records(
        _routing_records(dataset, agreement=agreement),
        dataset,
        ["q1"],
        ["dat_fixed", "dat"],
    )
    assert checked["dat"]["ndcg@10"] == checked["dat_fixed"]["ndcg@10"] == 1


@pytest.mark.parametrize(
    ("method", "agreement", "changes"),
    [
        ("dat_fixed", False, {"selected_alpha_dense": 0.5}),
        ("dat_fixed", False, {"selected_alpha_dense": None}),
        ("dat_fixed", False, {"selected_mode": None}),
        ("dat_fixed", False, {"selected_mode": "fixed_top1_agreement"}),
        ("dat_fixed", False, {"normalization": "full_raw"}),
        ("dat_fixed", False, {"top1_document_ids_dense_sparse": None}),
        ("dat_fixed", False, {"judge_probabilities_dense_sparse": [0.2, 0.8]}),
        ("dat", True, {"selected_alpha_dense": 0.5}),
        ("dat", True, {"selected_alpha_dense": None}),
        ("dat", True, {"selected_mode": "fixed"}),
        ("dat", True, {"selected_mode": None}),
        ("dat", True, {"normalization": "full_raw"}),
        (
            "dat",
            False,
            {"selected_mode": "fixed", "judge_probabilities_dense_sparse": None},
        ),
        (
            "dat",
            False,
            {"selected_mode": None, "judge_probabilities_dense_sparse": None},
        ),
        ("dat", False, {"selected_alpha_dense": 1 / 17 + 1e-11}),
        ("dat", False, {"normalization": "full_raw"}),
        ("dat", False, {"judge_probabilities_dense_sparse": None}),
    ],
)
def test_routing_rejects_invalid_policy_alpha_mode_and_probabilities(
    dataset, method, agreement, changes
):
    from evaluation.contextual_report import validate_records

    rows = _routing_records(dataset, agreement=agreement)
    next(row for row in rows if row["method"] == method).update(changes)
    with pytest.raises(ValueError):
        validate_records(rows, dataset, ["q1"], ["dat_fixed", "dat"])


def test_routing_rejects_mismatched_channel_top1_for_the_same_query(dataset):
    from evaluation.contextual_report import validate_records

    rows = _routing_records(dataset)
    rows[0]["top1_document_ids_dense_sparse"] = ["b", "a"]
    with pytest.raises(ValueError, match="top1"):
        validate_records(rows, dataset, ["q1"], ["dat_fixed", "dat"])


def test_agreement_requires_identical_top10_even_when_metrics_are_valid(dataset):
    from evaluation.contextual_report import ranking_record, validate_records

    rows = _routing_records(dataset, agreement=True)
    rows[1] = ranking_record(
        dataset,
        "q1",
        [0, 1],
        run_id="live",
        method="dat",
        alpha=0.84,
        judged=False,
        top1=["b", "b"],
    )
    with pytest.raises(ValueError, match="Top10"):
        validate_records(rows, dataset, ["q1"], ["dat_fixed", "dat"])


def test_records_recompute_metrics_and_keep_dense_weight_orientation(dataset):
    from evaluation.contextual_report import ranking_record

    row = ranking_record(
        dataset,
        "q1",
        [0, 1],
        run_id="fixture",
        method="dat",
        alpha=1 / 17,
        judged=True,
        top1=["a", "b"],
        probabilities=[0.2, 0.8],
    )
    assert row["p@1"] == 0
    assert row["recall@10"] == 1
    assert row["mrr@10"] == 0.5
    assert row["ndcg@10"] == pytest.approx(0.6309297535714575)
    assert row["relevant_rank_top10"] == 2
    assert row["top10_document_ids"] == ["a", "b"]
    assert row["selected_alpha_dense"] == pytest.approx(1 / 17)
    assert row["selected_mode"] == "judge_top1"
    assert row["oracle_alpha_dense"] is None


@pytest.mark.parametrize("ranking", [[0, 0], [-1], [2], [0.5]])
def test_invalid_rankings_cannot_be_exported(dataset, ranking):
    from evaluation.contextual_report import ranking_record

    with pytest.raises(ValueError, match="ranking"):
        ranking_record(dataset, "q1", ranking, run_id="fixture", method="dat")


def test_agreement_has_fixed_fallback_metadata_without_fake_probabilities(dataset):
    from evaluation.contextual_report import ranking_record

    row = ranking_record(
        dataset,
        "q1",
        [1, 0],
        run_id="fixture",
        method="dat",
        alpha=0.84,
        judged=False,
        top1=["b", "b"],
    )
    assert row["selected_mode"] == "fixed_top1_agreement"
    assert row["judge_probabilities_dense_sparse"] is None
    assert row["p@1"] == row["recall@10"] == row["mrr@10"] == row["ndcg@10"] == 1


def test_report_verification_rejects_tampered_metrics_and_missing_queries(dataset):
    from evaluation.contextual_report import validate_records

    records = [_routing_records(dataset, agreement=True)[1]]
    assert validate_records(records, dataset, ["q1"], ["dat"])["dat"]["ndcg@10"] == 1
    records[0]["ndcg@10"] = 0.9
    with pytest.raises(ValueError, match="metric"):
        validate_records(records, dataset, ["q1"], ["dat"])
    with pytest.raises(ValueError, match="coverage"):
        validate_records([], dataset, ["q1"], ["dat"])


def test_runs_are_unique_per_method_and_bound_to_expected_ids(dataset):
    from evaluation.contextual_report import validate_records

    row = _routing_records(dataset, agreement=True)[1]
    fixed = {**row, "method": "score_fusion", "run_id": "canonical"}
    expected = {"dat": "live", "score_fusion": "canonical"}
    assert set(
        validate_records(
            [row, fixed], dataset, ["q1"], list(expected), method_runs=expected
        )
    ) == set(expected)
    with pytest.raises(ValueError, match="duplicate"):
        validate_records([row, row], dataset, ["q1"], ["dat"])
    with pytest.raises(ValueError, match="run"):
        validate_records(
            [row, {**fixed, "run_id": "live"}],
            dataset,
            ["q1"],
            list(expected),
            method_runs=expected,
        )
    with pytest.raises(ValueError, match="run"):
        validate_records([{**row, "run_id": ""}], dataset, ["q1"], ["dat"])


def test_saved_report_verifies_without_models_and_rejects_identity_change(
    tmp_path, score_inputs
):
    from evaluation.contextual_report import (
        ranking_record,
        save_handoff,
        verify_handoff,
    )
    from evaluation.data import load_evaluation_dataset

    data_path = score_inputs / "data/processed/bccard"
    data = load_evaluation_dataset(data_path, "test", verify_all_splits=False)
    records = [
        ranking_record(
            data,
            q.query_id,
            [i, 1 - i],
            run_id="fixture",
            method="dat",
            alpha=0.84,
            judged=False,
            top1=[data.documents[i].document_id] * 2,
        )
        for i, q in enumerate(data.queries)
    ]
    output = tmp_path / "handoff"
    save_handoff(
        output,
        data_path,
        records,
        selection={"alpha": 0.84, "normalization": "union_minmax"},
        protocol={"scope": "test"},
    )
    assert verify_handoff(data_path, output)["queries"] == 2
    p = output / "per-query.jsonl"
    row = json.loads(p.read_text().splitlines()[0])
    row["ndcg@10"] = 0
    p.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="hash"):
        verify_handoff(data_path, output)


def test_live_runner_saves_observed_alpha_and_never_calls_judge_on_agreement(dataset):
    from evaluation.contextual_execution import run_queries
    from fusion.contextual import ContextualRetriever

    retrievers = {
        "dat_fixed": ContextualRetriever(
            dataset.documents,
            np.eye(2),
            encode=lambda q: [1, 0],
            sparse_scores=lambda q: [0, 1],
            judge=None,
            adaptive=False,
        ),
        "dat": ContextualRetriever(
            dataset.documents,
            np.eye(2),
            encode=lambda q: [1, 0],
            sparse_scores=lambda q: [0, 1],
            judge=lambda q, texts: [0.2, 0.8],
        ),
    }
    records, samples = run_queries(
        dataset, ["q1"], retrievers, run_id="fixture", repeats=2
    )
    adaptive = next(r for r in records if r["method"] == "dat")
    assert adaptive["selected_alpha_dense"] == pytest.approx(1 / 17)
    assert adaptive["top10_document_ids"] == ["b", "a"]
    assert len(samples) == 4
    assert all(s["total"] >= 0 for s in samples)


@pytest.mark.parametrize(
    ("split", "encoding_hash", "error"),
    [
        ("test", None, "selected on Validation"),
        (None, None, "selected on Validation"),
        ("validation", "different", "encoding"),
        ("validation", "matching", "snapshot reached"),
        ("validation", None, "snapshot reached"),
    ],
)
def test_live_evaluation_validates_policy_provenance(
    tmp_path, score_inputs, monkeypatch, split, encoding_hash, error
):
    import hashlib
    from types import SimpleNamespace

    pytest.importorskip("sentence_transformers")
    from evaluation import contextual_execution
    from evaluation.data import file_sha256

    data = score_inputs / "data/processed/bccard"
    cache = score_inputs / "artifacts/bccard-adaptive-vectors-resumed"
    encoding_path = cache / "encoding.json"
    encoding = json.loads(encoding_path.read_text())
    encoding["document_ids_sha256"] = hashlib.sha256(b"a\nb").hexdigest()
    encoding_path.write_text(json.dumps(encoding))
    hashes = {"dataset_manifest": file_sha256(data / "manifest.json")}
    if encoding_hash is not None:
        hashes["encoding"] = (
            file_sha256(encoding_path) if encoding_hash == "matching" else "0" * 64
        )
    policy = {"alpha": 0.84, "normalization": "full_raw", "input_sha256": hashes}
    if split is not None:
        policy["selection_split"] = split
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    args = SimpleNamespace(
        output=tmp_path / "run",
        data=data,
        vector_cache=cache,
        policy=policy_path,
        encoder_name="BAAI/bge-m3",
        encoder_revision="1" * 40,
        encoder_model=tmp_path / "model",
    )

    def stop_before_loading_models(path):
        raise ValueError("snapshot reached")

    monkeypatch.setattr(
        contextual_execution, "snapshot_identity", stop_before_loading_models
    )
    with pytest.raises(ValueError, match=error):
        contextual_execution.live_evaluate(args)


def test_judge_prompt_preserves_suffix_when_document_is_truncated():
    from fusion.contextual_judge import judge_inputs

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text.encode())

    sequences, truncated = judge_inputs(
        Tokenizer(), "query", ["x" * 4000, "body"], max_tokens=512
    )
    assert len(sequences[0]["input_ids"]) == 512
    assert truncated == 1
    assert bytes(sequences[0]["input_ids"]).endswith(b"</think>\n\n")
    assert b"<Query>: query" in bytes(sequences[1]["input_ids"])
    with pytest.raises(ValueError, match="budget"):
        judge_inputs(Tokenizer(), "query", ["body"], max_tokens=2)


def test_cli_rejects_latency_without_measured_pass_before_loading_models(tmp_path):
    from scripts.evaluate_contextual import main

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "evaluate",
                "--data",
                str(tmp_path),
                "--output",
                str(tmp_path / "run"),
                "--repeats",
                "1",
            ]
        )
    assert exc.value.code == 2


def test_fresh_selection_uses_validation_without_reading_test_qrels(
    score_inputs, tmp_path
):
    import hashlib

    from evaluation.contextual import select_vector_policy

    data = score_inputs / "data/processed/bccard"
    cache = score_inputs / "artifacts/bccard-adaptive-vectors-resumed"
    path = cache / "encoding.json"
    encoding = json.loads(path.read_text())
    encoding["document_ids_sha256"] = hashlib.sha256(b"a\nb").hexdigest()
    path.write_text(json.dumps(encoding))
    (data / "qrels_test.jsonl").write_text("broken and must never be read")
    selected = select_vector_policy(data, cache, tmp_path / "selected.json")
    assert selected["validation_queries"] == 2
    assert selected["selection_split"] == "validation"
    assert selected["alpha"] == 0
    assert selected["normalization"] == "full_raw"


def test_timing_audit_recomputes_measured_pass_and_rejects_stage_mismatch():
    from evaluation.contextual_report import validate_timings

    samples = [
        {
            "query_id": "q",
            "method": method,
            "repeat": rep,
            "encode": total - 3,
            "retrieve": 1.0,
            "judge": 1.0,
            "fusion": 1.0,
            "total": total,
        }
        for rep, method, total in [
            (0, "dat_fixed", 5.0),
            (0, "dat", 7.0),
            (1, "dat_fixed", 3.0),
            (1, "dat", 5.0),
        ]
    ]
    summary = validate_timings(samples, ["q"], ["dat_fixed", "dat"], repeats=2)
    assert summary["dat_fixed"]["total"]["mean_ms"] == 3
    assert summary["dat"]["total"]["mean_ms"] == 5
    samples[-1]["total"] = 9
    with pytest.raises(ValueError, match="stage"):
        validate_timings(samples, ["q"], ["dat_fixed", "dat"], repeats=2)


def test_timing_audit_rejects_missing_and_duplicate_requests():
    from evaluation.contextual_report import validate_timings

    row = {
        "query_id": "q",
        "method": "dat",
        "repeat": 0,
        "encode": 1.0,
        "retrieve": 1.0,
        "judge": 1.0,
        "fusion": 1.0,
        "total": 4.0,
    }
    with pytest.raises(ValueError, match="coverage"):
        validate_timings([row], ["q"], ["dat"], repeats=2)
    with pytest.raises(ValueError, match="duplicate"):
        validate_timings([row, row], ["q"], ["dat"], repeats=2)


@pytest.mark.parametrize("weight_name", ["pytorch_model.bin", "model.safetensors"])
def test_snapshot_identity_supports_existing_pinned_weight_formats(
    tmp_path, weight_name
):
    from evaluation.contextual_execution import snapshot_identity
    from evaluation.data import file_sha256

    (tmp_path / weight_name).write_bytes(b"fixture weights; no inference")
    result = snapshot_identity(tmp_path)
    assert result["files_sha256"][weight_name] == file_sha256(tmp_path / weight_name)
    before = result["snapshot_sha256"]
    (tmp_path / weight_name).write_bytes(b"changed weights")
    assert snapshot_identity(tmp_path)["snapshot_sha256"] != before


def test_snapshot_in_huggingface_cache_keeps_weights_but_ignores_internal_cache(
    tmp_path,
):
    from evaluation.contextual_execution import snapshot_identity

    root = tmp_path / ".cache" / "snapshot"
    root.mkdir(parents=True)
    (root / "pytorch_model.bin").write_bytes(b"weights")
    (root / ".cache").mkdir()
    (root / ".cache" / "download.lock").touch()
    assert list(snapshot_identity(root)["files_sha256"]) == ["pytorch_model.bin"]


def test_cli_limits_numpy_loaded_in_a_fresh_process(score_inputs, tmp_path):
    import subprocess
    import sys

    from evaluation.contextual_report import ranking_record, save_handoff
    from evaluation.data import load_evaluation_dataset

    data_path = score_inputs / "data/processed/bccard"
    data = load_evaluation_dataset(data_path, "test", verify_all_splits=False)
    records = [
        ranking_record(data, q.query_id, [i, 1 - i], run_id="fixture", method="stub")
        for i, q in enumerate(data.queries)
    ]
    output = tmp_path / "report"
    save_handoff(output, data_path, records, selection={}, protocol={})
    code = """
import json, sys
from scripts.evaluate_contextual import main
from threadpoolctl import threadpool_info
def profile(frame, event, arg):
    if event == "call" and frame.f_code.co_name == "verify_handoff":
        print("POOLS=" + json.dumps(threadpool_info()))
sys.setprofile(profile)
main(sys.argv[1:])
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "verify",
            "--data",
            str(data_path),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    pools = json.loads(
        next(
            s.removeprefix("POOLS=")
            for s in result.stdout.splitlines()
            if s.startswith("POOLS=")
        )
    )
    blas = [p for p in pools if p["user_api"] == "blas"]
    assert blas and all(p["num_threads"] == 2 for p in blas)


@pytest.fixture
def combined_handoff(tmp_path, score_inputs):
    from evaluation.contextual_report import (
        combine_handoffs,
        ranking_record,
        save_handoff,
        validate_records,
    )
    from evaluation.data import load_evaluation_dataset

    data_path = score_inputs / "data/processed/bccard"
    data = load_evaluation_dataset(data_path, "test", verify_all_splits=False)
    baseline_methods = ("bm25", "dense", "score_fusion")
    live_methods = ("dat_fixed", "dat")
    outputs = []
    for name, methods in (("baseline", baseline_methods), ("live", live_methods)):
        records = [
            ranking_record(
                data,
                q.query_id,
                [1 - i, i] if method in live_methods else [i, 1 - i],
                run_id=name,
                method=method,
                alpha=0.84 if method in live_methods else None,
                judged=False if method == "dat" else None,
                top1=[data.documents[1 - i].document_id] * 2
                if method in live_methods
                else None,
            )
            for i, q in enumerate(data.queries)
            for method in methods
        ]
        output = tmp_path / name
        samples = (
            [
                dict(
                    query_id=q.query_id,
                    method=m,
                    repeat=r,
                    encode=1.0,
                    retrieve=2.0,
                    judge=3.0,
                    fusion=4.0,
                    total=10.0,
                )
                for q in data.queries
                for m in live_methods
                for r in range(3)
            ]
            if name == "live"
            else None
        )
        save_handoff(
            output,
            data_path,
            records,
            selection={"alpha": 0.84, "normalization": "union_minmax"}
            if name == "live"
            else {},
            protocol={},
            samples=samples,
            environment={"passes": 3} if samples else None,
            baseline_audit={
                "all_three_main_baselines_reproduced": True,
                "queries": len(data.queries),
                "canonical_metrics": validate_records(
                    records, data, [q.query_id for q in data.queries], list(methods)
                ),
                "replayed_metrics": validate_records(
                    records, data, [q.query_id for q in data.queries], list(methods)
                ),
            }
            if name == "baseline"
            else None,
        )
        outputs.append(output)
    output = tmp_path / "combined"
    combine_handoffs(data_path, *outputs, output)
    return data_path, output


def _refresh_artifact_hash(output, name):
    from evaluation.data import file_sha256

    protocol = json.loads((output / "protocol.json").read_text())
    protocol["artifact_sha256"][name] = file_sha256(output / name)
    (output / "protocol.json").write_text(json.dumps(protocol))


def test_combined_handoff_preserves_runs_and_recomputes_rankings_and_latency(
    combined_handoff,
):
    from evaluation.contextual_report import verify_handoff

    data, output = combined_handoff
    checked = verify_handoff(data, output)
    assert checked["rows"] == 10
    assert checked["timing_samples"] == 12
    assert set(p.name for p in output.iterdir()) == {
        "protocol.json",
        "results.json",
        "per-query.jsonl",
        "latency-samples.json",
    }
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["method_runs"]["score_fusion"] == "baseline"
    assert protocol["method_runs"]["dat_fixed"] == "live"
    assert "query_ids" not in protocol
    assert checked["latency"]["dat"]["total"]["mean_ms"] == 10
    assert checked["methods"]["score_fusion"]["ndcg@10"] == 1
    assert checked["methods"]["dat_fixed"]["ndcg@10"] == pytest.approx(
        0.6309297535714575
    )


@pytest.mark.parametrize(
    "change", ["mixed_run", "missing", "duplicate", "metric", "query_order"]
)
def test_combined_verification_rejects_invalid_rows_even_with_updated_hash(
    combined_handoff, change
):
    from evaluation.contextual_report import verify_handoff

    data, output = combined_handoff
    path = output / "per-query.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    if change == "mixed_run":
        rows[-1]["run_id"] = "another-live-run"
    elif change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[-2]
    elif change == "metric":
        rows[-1]["ndcg@10"] = 0.9
    else:
        rows.reverse()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_artifact_hash(output, path.name)
    with pytest.raises(ValueError):
        verify_handoff(data, output)


@pytest.mark.parametrize(
    "change", ["missing_sample", "stage_sum", "summary", "missing_hash", "timing_run"]
)
def test_combined_verification_rejects_invalid_timing(combined_handoff, change):
    from evaluation.contextual_report import verify_handoff

    data, output = combined_handoff
    path = output / "latency-samples.json"
    samples = json.loads(path.read_text())
    if change == "missing_sample":
        samples.pop()
    elif change == "stage_sum":
        samples[-1]["total"] = 99
    elif change == "summary":
        result_path = output / "results.json"
        results = json.loads(result_path.read_text())
        results["latency"]["dat"]["total"]["mean_ms"] = 99
        result_path.write_text(json.dumps(results))
        _refresh_artifact_hash(output, result_path.name)
    else:
        protocol_path = output / "protocol.json"
        protocol = json.loads(protocol_path.read_text())
        if change == "missing_hash":
            protocol["artifact_sha256"].pop(path.name)
        else:
            protocol["timing"]["method_runs"]["dat"] = "baseline"
        protocol_path.write_text(json.dumps(protocol))
    path.write_text(json.dumps(samples))
    if change != "missing_hash":
        _refresh_artifact_hash(output, path.name)
    with pytest.raises(ValueError):
        verify_handoff(data, output)


def test_compact_encoding_keeps_settings_and_hashes_instead_of_query_arrays():
    from evaluation.contextual_report import compact_encoding, ordered_ids_sha256

    encoding = {
        "model": {"revision": "pinned"},
        "vector_sha256": {"document-vectors.npy": "hash"},
        "splits": {
            "train": {
                "query_ids": ["q1", "q2"],
                "query_encoding_latency": {"mean_ms": 1.0},
            }
        },
    }
    compact = compact_encoding(encoding)
    assert encoding["splits"]["train"]["query_ids"] == ["q1", "q2"]
    assert compact["splits"]["train"] == {
        "queries": 2,
        "query_ids_sha256": ordered_ids_sha256(["q1", "q2"]),
        "query_encoding_latency": {"mean_ms": 1.0},
    }
    assert compact["model"] == encoding["model"]
    assert compact["vector_sha256"] == encoding["vector_sha256"]
    assert compact_encoding(compact) == compact


@pytest.mark.parametrize("field", ["canonical_metrics", "replayed_metrics"])
def test_combined_verification_rejects_changed_baseline_audit(combined_handoff, field):
    from evaluation.contextual_report import verify_handoff

    data, output = combined_handoff
    path = output / "results.json"
    results = json.loads(path.read_text())
    results["baseline_audit"][field]["score_fusion"]["ndcg@10"] = 0.9
    path.write_text(json.dumps(results))
    _refresh_artifact_hash(output, path.name)
    with pytest.raises(ValueError, match="baseline"):
        verify_handoff(data, output)


def test_combined_verification_rejects_changed_dataset(combined_handoff):
    from evaluation.contextual_report import verify_handoff

    data, output = combined_handoff
    path = data / "qrels_test.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="dataset hash"):
        verify_handoff(data, output)


@pytest.mark.parametrize("change", ["alpha", "normalization", "missing_alpha"])
def test_verification_binds_fixed_rows_to_the_stored_selection(
    combined_handoff, change
):
    from evaluation.contextual_report import verify_handoff

    data, output = combined_handoff
    path = output / "protocol.json"
    protocol = json.loads(path.read_text())
    if change == "missing_alpha":
        protocol["selection"].pop("alpha")
    else:
        protocol["selection"][change] = 0.6 if change == "alpha" else "full_raw"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="Fixed policy"):
        verify_handoff(data, output)


def test_verification_accepts_a_different_validation_selected_fixed_alpha(
    combined_handoff,
):
    from evaluation.contextual_report import verify_handoff

    data, output = combined_handoff
    path = output / "per-query.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for row in records:
        if row["method"] in ("dat_fixed", "dat"):
            row["selected_alpha_dense"] = 0.6
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    _refresh_artifact_hash(output, path.name)
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["selection"]["alpha"] = 0.6
    protocol_path.write_text(json.dumps(protocol))
    assert verify_handoff(data, output)["verified"] is True


@pytest.mark.parametrize(
    ("method", "changes", "error"),
    [
        ("dat_fixed", {"selected_alpha_dense": 0.5}, "Fixed policy"),
        ("dat", {"selected_alpha_dense": 0.5}, "Fixed policy"),
        ("dat", {"selected_mode": "fixed"}, "selected mode"),
        ("dat_fixed", {"top1_document_ids_dense_sparse": ["a", "a"]}, "top1"),
    ],
)
def test_verification_rejects_routing_tampering_even_with_updated_artifact_hash(
    combined_handoff, method, changes, error
):
    from evaluation.contextual_report import verify_handoff

    data, output = combined_handoff
    path = output / "per-query.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    next(row for row in records if row["method"] == method).update(changes)
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    _refresh_artifact_hash(output, path.name)
    with pytest.raises(ValueError, match=error):
        verify_handoff(data, output)


def test_one_method_cannot_mix_runs_without_an_explicit_mapping(dataset):
    from dataclasses import replace

    from evaluation.contextual_report import ranking_record, validate_records

    data = replace(
        dataset,
        queries=(*dataset.queries, Query("q2", "another")),
        qrels={**dataset.qrels, "q2": {"a": 3}},
    )
    records = [
        ranking_record(
            data,
            q,
            [0, 1],
            run_id=run,
            method="dat",
            alpha=0.84,
            judged=False,
            top1=["a", "a"],
        )
        for q, run in (("q1", "first"), ("q2", "second"))
    ]
    with pytest.raises(ValueError, match="mixed run"):
        validate_records(records, data, ["q1", "q2"], ["dat"])

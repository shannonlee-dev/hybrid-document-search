"""Enterprise export preserves primary rankings and satisfies Phase 5 report inputs."""

import copy
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")


@pytest.fixture
def handoff(score_inputs):
    from evaluation.qdap_benchmark import sha

    directory = score_inputs / "handoff"
    directory.mkdir()
    source = directory / "per-query.jsonl"
    records = []
    for i in (0, 1):
        records.append(
            dict(
                run_id="frozen-run",
                query_id=f"test-{i}",
                method="QDAP-S",
                **{
                    "p@1": 1.0 if i == 0 else 0.0,
                    "recall@10": 1.0,
                    "mrr@10": 1.0 if i == 0 else 0.5,
                    "ndcg@10": 1.0 if i == 0 else 0.6309297535714575,
                },
                top10_document_ids=["a", "b"],
                relevant_rank_top10=i + 1,
                selected_alpha_dense=0.2 if i == 0 else 0.8,
                oracle_alpha_dense=0.0,
                selected_mode=None,
                bm25_weight=None,
            )
        )
    source.write_text("".join(json.dumps(r) + "\n" for r in records))
    selection = directory / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selected": {
                    "S": {
                        "seed": 42,
                        "epoch": 4,
                        "readout": "argmax",
                        "ce_mode": "literal",
                        "checkpoint_sha256": "a" * 64,
                    }
                }
            }
        )
    )
    data = score_inputs / "data/processed/bccard"
    manifest = {
        "sha256": sha(source),
        "selection_sha256": sha(selection),
        "input_sha256": {
            str(p): sha(p)
            for p in [
                data / "manifest.json",
                data / "corpus.jsonl",
                data / "queries_test.jsonl",
                data / "qrels_test.jsonl",
            ]
        },
    }
    encoding = score_inputs / "artifacts/bccard-adaptive-vectors-resumed/encoding.json"
    record = json.loads(encoding.read_text())
    manifest["encoding"] = {
        key: record[key]
        for key in (
            "model",
            "dtype",
            "max_seq_length",
            "vector_sha256",
            "dataset_manifest_sha256",
        )
    }
    manifest["encoding_record_sha256"] = sha(encoding)
    (directory / "per-query-manifest.json").write_text(json.dumps(manifest))
    (directory / "protocol.json").write_text(
        json.dumps(
            {
                "input_sha256": {
                    str(encoding): sha(encoding),
                    **{
                        str(encoding.parent / name): digest
                        for name, digest in record["vector_sha256"].items()
                    },
                }
            }
        )
    )
    # Main's saved replay is a separate run, with independently pinned metrics.
    canonical_rows = [
        row | {"run_id": "main-replay", "method": method}
        for method in ("BM25", "Dense", "Existing Fixed")
        for row in records
    ]
    canonical_source = directory / "main-baseline-per-query.jsonl"
    canonical_source.write_text("".join(json.dumps(r) + "\n" for r in canonical_rows))
    summary = directory / "canonical-summary.json"
    summary.write_text(
        json.dumps(
            {
                "dataset": {
                    "split": "test",
                    "input_sha256": {
                        Path(p).name: digest
                        for p, digest in manifest["input_sha256"].items()
                        if Path(p).name != "manifest.json"
                    },
                },
                "encoding": manifest["encoding"],
                "methods": {
                    method: {
                        "config": {},
                        "slices": {
                            "non_synthetic": {
                                "query_count": 2,
                                "p@1": 0.5,
                                "recall@10": 1.0,
                                "mrr@10": 0.75,
                                "ndcg@10": 0.8154648767857288,
                            }
                        },
                    }
                    for method in ("bm25", "dense", "score_fusion")
                },
            }
        )
    )
    (directory / "baseline-audit.json").write_text(
        json.dumps(
            {
                "run_id": "main-replay",
                "canonical_summary": str(summary),
                "canonical_summary_sha256": sha(summary),
                "per_query_sha256": sha(canonical_source),
                "current_vector_sha256": record["vector_sha256"],
            }
        )
    )
    return score_inputs, source


def test_export_keeps_rankings_metrics_weights_and_identity(handoff):
    from evaluation.qdap_report import enterprise_report

    inputs, source = handoff
    original = source.read_bytes()
    report = enterprise_report(inputs, source)
    assert report["schema_version"] == "1"
    assert report["dataset"]["split"] == "test"
    assert report["dataset"]["query_ids"] == ["test-0", "test-1"]
    assert report["conditions"]["top_k"] == 10
    assert (
        report["conditions"]["relevance"]
        == "binary metrics use relevance > 0; graded nDCG uses gains 1-3"
    )
    assert report["encoding"]["model"]["revision"] == "1" * 40
    method = report["methods"]["qdap_s"]
    assert method["latency"] is None
    assert method["slices"]["non_synthetic"]["query_count"] == 2
    assert method["slices"]["non_synthetic"]["ndcg@10"] == pytest.approx(
        0.8154648767857288
    )
    row = method["queries"][1]
    assert row["query_id"] == "test-1"
    assert [r["document_id"] for r in row["results"]] == ["a", "b"]
    assert row["metrics"]["rr@10"] == row["metrics"]["mrr@10"] == 0.5
    assert row["metrics"]["graded_ndcg@10"] == pytest.approx(0.6309297535714575)
    assert row["selected_alpha_dense"] == 0.8
    assert all(r["score"] is None for r in row["results"])
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "metric",
        "dataset",
        "corpus",
        "query",
        "qrels",
        "encoding",
        "nonfinite",
    ],
)
def test_export_rejects_corrupt_or_misaligned_handoff(handoff, kind):
    from evaluation.qdap_benchmark import sha
    from evaluation.qdap_report import enterprise_report

    inputs, source = handoff
    if kind == "encoding":
        p = source.parent / "per-query-manifest.json"
        manifest = json.loads(p.read_text())
        manifest["encoding"]["vector_sha256"]["document-vectors.npy"] = "b" * 64
        p.write_text(json.dumps(manifest))
    elif kind in ("corpus", "query", "qrels"):
        names = {
            "corpus": "corpus.jsonl",
            "query": "queries_test.jsonl",
            "qrels": "qrels_test.jsonl",
        }
        p = inputs / "data/processed/bccard" / names[kind]
        p.write_text(p.read_text() + "\n")
    elif kind == "dataset":
        p = inputs / "data/processed/bccard/manifest.json"
        p.write_text(p.read_text() + " ")
    else:
        records = [json.loads(line) for line in source.read_text().splitlines()]
        if kind == "duplicate":
            records.append(records[0])
        elif kind == "nonfinite":
            records[0]["ndcg@10"] = float("nan")
        else:
            records[0]["ndcg@10"] = 0
        source.write_text("".join(json.dumps(r) + "\n" for r in records))
        p = source.parent / "per-query-manifest.json"
        manifest = json.loads(p.read_text())
        manifest["sha256"] = sha(source)
        p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        enterprise_report(inputs, source)


@pytest.mark.parametrize("local_cache", ["missing", "different"])
def test_export_uses_original_saved_identity_without_local_vectors(
    handoff, local_cache
):
    import shutil

    from evaluation.qdap_report import enterprise_report

    inputs, source = handoff
    expected = json.loads((source.parent / "per-query-manifest.json").read_text())[
        "encoding"
    ]
    cache = inputs / "artifacts/bccard-adaptive-vectors-resumed"
    if local_cache == "missing":
        shutil.rmtree(inputs / "artifacts")
    else:
        (cache / "encoding.json").write_text(
            json.dumps(
                {
                    "model": {"model_name": "other", "revision": "f" * 40},
                    "vector_sha256": {"document-vectors.npy": "f" * 64},
                }
            )
        )
        (cache / "document-vectors.npy").write_bytes(b"different re-encoding")
    original = source.read_bytes()
    report = enterprise_report(inputs, source)
    assert report["encoding"]["model"] == expected["model"]
    assert report["encoding"]["vector_sha256"] == expected["vector_sha256"]
    assert report["methods"]["qdap_s"]["queries"][1]["metrics"]["mrr@10"] == 0.5
    assert source.read_bytes() == original


def test_reference_alignment_removes_extra_queries_without_changing_source(handoff):
    from evaluation.qdap_report import align_reference_report, enterprise_report

    inputs, source = handoff
    target = enterprise_report(inputs, source)
    reference = copy.deepcopy(target)
    method = reference["methods"]["qdap_s"]
    method["queries"].append(
        dict(
            query_id="synthetic-extra",
            metrics={k: 0 for k in method["queries"][0]["metrics"]},
            results=[],
        )
    )
    reference["dataset"]["query_count"] = 3
    original = copy.deepcopy(reference)
    aligned = align_reference_report(reference, target)
    assert reference == original
    assert [r["query_id"] for r in aligned["methods"]["qdap_s"]["queries"]] == [
        "test-0",
        "test-1",
    ]
    assert aligned["dataset"]["query_count"] == 2
    assert aligned["methods"]["qdap_s"]["slices"]["all"]["query_count"] == 2
    reference["dataset"]["manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="identity"):
        align_reference_report(reference, target)


def test_enterprise_export_cli_needs_no_training_run(handoff, tmp_path):
    import os
    import shutil
    import subprocess
    import sys

    inputs, source = handoff
    shutil.rmtree(inputs / "artifacts")
    output = tmp_path / "reports"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.export_qdap_results",
            "--stage",
            "enterprise",
            "--inputs",
            str(inputs),
            "--per-query",
            str(source),
            "--output",
            str(output),
            "--canonical-summary",
            str(source.parent / "canonical-summary.json"),
        ],
        env=os.environ | {"PYTHONPATH": str(Path.cwd())},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["methods"]["qdap_s"]["queries"][0]["query_id"] == "test-0"


def test_export_separates_internal_dense_from_official_baselines(handoff):
    from evaluation.qdap_benchmark import sha
    from evaluation.qdap_report import enterprise_report

    inputs, source = handoff
    records = [json.loads(line) for line in source.read_text().splitlines()]
    records += [row | {"method": "Dense"} for row in records]
    source.write_text("".join(json.dumps(row) + "\n" for row in records))
    path = source.parent / "per-query-manifest.json"
    manifest = json.loads(path.read_text()) | {"sha256": sha(source)}
    path.write_text(json.dumps(manifest))
    report = enterprise_report(inputs, source)
    assert report["conditions"]["official_baselines"] == [
        "bm25",
        "dense",
        "score_fusion",
    ]
    assert report["methods"]["dense"]["config"]["source_run_id"] == "main-replay"
    internal = report["methods"]["qdap_dense_fp64"]
    assert internal["config"]["baseline_role"] == "qdap_protocol"
    assert internal["config"]["source_run_id"] == "frozen-run"
    assert report["methods"]["dense"]["config"]["baseline_role"] == "main_canonical"


@pytest.mark.parametrize("corruption", ["missing", "hash", "query_ids", "metrics"])
def test_export_rejects_invalid_canonical_baseline(handoff, corruption):
    from evaluation.qdap_benchmark import sha
    from evaluation.qdap_report import enterprise_report

    inputs, source = handoff
    canonical = source.parent / "main-baseline-per-query.jsonl"
    audit_path = source.parent / "baseline-audit.json"
    audit = json.loads(audit_path.read_text())
    if corruption == "missing":
        canonical.unlink()
    elif corruption == "hash":
        canonical.write_text(canonical.read_text() + "\n")
    elif corruption == "query_ids":
        records = [json.loads(line) for line in canonical.read_text().splitlines()]
        canonical.write_text("".join(json.dumps(r) + "\n" for r in records[1:]))
        audit["per_query_sha256"] = sha(canonical)
    else:
        summary_path = source.parent / "canonical-summary.json"
        summary = json.loads(summary_path.read_text())
        summary["methods"]["dense"]["slices"]["non_synthetic"]["ndcg@10"] = 0.1
        summary_path.write_text(json.dumps(summary))
        audit["canonical_summary_sha256"] = sha(summary_path)
    audit_path.write_text(json.dumps(audit))
    with pytest.raises((ValueError, FileNotFoundError)):
        enterprise_report(inputs, source)


def test_replay_records_frozen_and_current_source_hashes(handoff):
    from evaluation.qdap_benchmark import sha
    from evaluation.qdap_report import enterprise_report

    inputs, source = handoff
    selection_path = source.parent / "selection.json"
    selection = json.loads(selection_path.read_text())
    selection["source_sha256"] = {"fusion/qdap.py": "0" * 64}
    selection_path.write_text(json.dumps(selection))
    manifest_path = source.parent / "per-query-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selection_sha256"] = sha(selection_path)
    manifest_path.write_text(json.dumps(manifest))
    original = selection_path.read_bytes()
    report = enterprise_report(inputs, source)
    provenance = report["source"]["code_provenance"]
    assert provenance["frozen_experiment_source_sha256"] == selection["source_sha256"]
    assert provenance["current_source_sha256"]["fusion/qdap.py"] == sha(
        "fusion/qdap.py"
    )
    assert provenance["source_hash_mismatches"] == ["fusion/qdap.py"]
    assert selection_path.read_bytes() == original


def test_official_721_export_preserves_canonical_and_all_qdap_rankings():
    from evaluation.qdap_benchmark import sha
    from evaluation.qdap_report import enterprise_report

    root = Path(__file__).resolve().parents[2]
    inputs = Path(os.environ.get("QDAP_INPUTS", root))
    if not (inputs / "data/processed/bccard/manifest.json").is_file():
        pytest.skip("Set QDAP_INPUTS to the prepared BC Card data root")
    directory = root / "docs/qdap-official-2026-10-30"
    source = directory / "per-query.jsonl"
    before = {p.name: sha(p) for p in directory.iterdir() if p.is_file()}
    report = enterprise_report(inputs, source)
    summary = json.loads(
        (root / "docs/experiments/bccard/test-summary.json").read_text()
    )
    ids = report["dataset"]["query_ids"]
    assert len(ids) == len(set(ids)) == 721
    manifest = json.loads((inputs / "data/processed/bccard/manifest.json").read_text())
    assert ids == manifest["primary_slice_query_ids"]
    metrics = ("p@1", "recall@10", "mrr@10", "ndcg@10")
    for name, ndcg in (
        ("bm25", 0.5755128098916078),
        ("dense", 0.7835091332711331),
        ("score_fusion", 0.7940145938588489),
    ):
        method = report["methods"][name]
        assert [q["query_id"] for q in method["queries"]] == ids
        actual = method["slices"]["non_synthetic"]
        assert actual["ndcg@10"] == pytest.approx(ndcg, abs=1e-12, rel=0)
        for metric in metrics:
            assert actual[metric] == pytest.approx(
                summary["methods"][name]["slices"]["non_synthetic"][metric],
                abs=1e-12,
                rel=0,
            )
    aliases = {
        "BM25": "qdap_bm25",
        "Dense": "qdap_dense_fp64",
        "Existing Fixed": "qdap_fixed_fp64",
        "Existing Router": "qdap_router_fp64",
        "Paper Fixed": "paper_fixed",
        "QDAP-S": "qdap_s",
        "QDAP-L": "qdap_l",
        "Oracle alpha": "oracle_alpha",
    }
    for filename, names in (
        ("per-query.jsonl", aliases),
        (
            "main-baseline-per-query.jsonl",
            {"BM25": "bm25", "Dense": "dense", "Existing Fixed": "score_fusion"},
        ),
    ):
        exported = {
            name: {q["query_id"]: q for q in report["methods"][alias]["queries"]}
            for name, alias in names.items()
        }
        for name in names:
            assert set(exported[name]) == set(ids)
        for row in map(json.loads, (directory / filename).read_text().splitlines()):
            actual = exported[row["method"]][row["query_id"]]
            assert [hit["document_id"] for hit in actual["results"]] == row[
                "top10_document_ids"
            ]
            for metric in metrics:
                assert actual["metrics"][metric] == row[metric]
    assert report["methods"]["qdap_dense_fp64"]["slices"]["all"][
        "ndcg@10"
    ] == pytest.approx(0.7828810770027878)
    assert report["methods"]["paper_fixed"]["config"]["alpha_dense"] == 0.84
    assert before == {p.name: sha(p) for p in directory.iterdir() if p.is_file()}

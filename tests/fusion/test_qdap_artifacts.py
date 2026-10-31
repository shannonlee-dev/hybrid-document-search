"""Portable identities validate the inputs consumed by a stage, never old paths."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.json_io import read_json, write_json
from evaluation.qdap_artifacts import (
    DATA,
    ROUTER,
    VECTORS,
    check,
    evaluation_files,
    hashes,
    model_identity,
    resolve_selection,
    sha,
)


@pytest.fixture
def portable_run(tmp_path):
    args = SimpleNamespace(
        inputs=tmp_path / "inputs",
        output=tmp_path / "run",
        model=tmp_path / "model",
        vector_cache=tmp_path / "vectors",
        model_name="BAAI/bge-m3",
        model_revision="1" * 40,
    )
    args.output.mkdir()
    args.model.mkdir()
    (args.model / "model.safetensors").write_bytes(b"weights")
    (args.model / "config.json").write_text("{}")
    for identity, path in evaluation_files(args).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(identity)
    write_json(args.output / "protocol.json", {})
    checkpoint = args.output / "checkpoints/s.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"selected")
    selection = {
        "schema_version": 2,
        "selection_split": "validation",
        "protocol_sha256": sha(args.output / "protocol.json"),
        "model_identity": model_identity(args),
        "evaluation_input_sha256": hashes(evaluation_files(args)),
        "selected": {
            "S": {
                "checkpoint": "checkpoints/s.pt",
                "checkpoint_sha256": sha(checkpoint),
                "epoch": 4,
                "readout": "argmax",
            }
        },
    }
    write_json(args.output / "selection.json", selection)
    return args


def test_relocation_preserves_content_identity_and_selection(portable_run, tmp_path):
    args = portable_run
    original = (args.output / "selection.json").read_bytes()
    for name in ("inputs", "output", "model", "vector_cache"):
        old = getattr(args, name)
        new = tmp_path / (name + "-moved")
        shutil.move(old, new)
        setattr(args, name, new)
    resolved = resolve_selection(args)
    assert resolved["selected"]["S"]["epoch"] == 4
    assert resolved["selected"]["S"]["readout"] == "argmax"
    assert Path(resolved["selected"]["S"]["checkpoint"]).is_file()
    assert (args.output / "selection.json").read_bytes() == original


@pytest.mark.parametrize(
    "identity",
    [
        "data/corpus.jsonl",
        "data/queries_test.jsonl",
        "data/qrels_test.jsonl",
        "data/manifest.json",
        "router/router.json",
        "vectors/document-vectors.npy",
        "vectors/query-vectors-test.npy",
        "vectors/encoding.json",
        "model/model.safetensors",
    ],
)
def test_consumed_input_mutation_is_rejected(portable_run, identity):
    path = evaluation_files(portable_run)[identity]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_selection(portable_run)


def test_missing_and_changed_checkpoint_are_explicit(portable_run, tmp_path):
    args = portable_run
    path = args.output / "checkpoints/s.pt"
    path.unlink()
    with pytest.raises(FileNotFoundError, match="missing S checkpoint"):
        resolve_selection(args)
    args.checkpoint_s = tmp_path / "different-location.pt"
    args.checkpoint_s.write_bytes(b"selected")
    resolve_selection(args)
    args.checkpoint_s.write_bytes(b"other epoch")
    with pytest.raises(ValueError, match="SHA-256 mismatch: S checkpoint"):
        resolve_selection(args)


def test_model_revision_mismatch(portable_run):
    portable_run.model_revision = "2" * 40
    with pytest.raises(ValueError, match="model revision mismatch"):
        resolve_selection(portable_run)


def test_legacy_read_only_with_only_consumed_dependencies(portable_run):
    args = portable_run
    selection = read_json(args.output / "selection.json")
    selection.pop("schema_version")
    selection.pop("model_identity")
    selection["model"] = "/previous/model"
    roots = {
        "data": Path("/previous/project") / DATA,
        "vectors": Path("/previous/project") / VECTORS,
        "router": Path("/previous/project") / ROUTER.parent,
        "model": Path("/previous/model"),
    }
    selection["evaluation_input_sha256"] = {
        str(roots[key.split("/")[0]] / key.split("/", 1)[1]): digest
        for key, digest in selection["evaluation_input_sha256"].items()
    }
    protocol = {
        "input_sha256": {
            key: selection["evaluation_input_sha256"].pop(key)
            for key in list(selection["evaluation_input_sha256"])
            if key.endswith("encoding.json")
        }
    }
    write_json(args.output / "protocol.json", protocol)
    selection["protocol_sha256"] = sha(args.output / "protocol.json")
    selection["evaluation_input_sha256"].update(
        {
            "/previous/project/artifacts/paper-guided/dat-local/bccard/" + name: "0"
            * 64
            for name in (
                "dense.npy",
                "sparse.npy",
                "queries.json",
                "unrelated-future-file",
            )
        }
    )
    selection["selected"]["S"]["checkpoint"] = "/previous/run/checkpoints/s.pt"
    write_json(args.output / "selection.json", selection)
    original = (args.output / "selection.json").read_bytes()
    args.model_revision = None
    resolve_selection(args)
    assert (args.output / "selection.json").read_bytes() == original


def test_digest_is_location_independent(tmp_path):
    path = tmp_path / "old"
    path.write_bytes(b"same")
    digest = sha(path)
    path.rename(tmp_path / "new")
    check(tmp_path / "new", digest, "input")
    (tmp_path / "new").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        check(tmp_path / "new", digest, "input")


def test_official_evidence_hashes_and_method_counts():
    """Always runs without private data: compare evidence to the immutable Git blob."""
    from collections import Counter

    root = Path(__file__).resolve().parents[2]
    directory = root / "docs/validation/2026-10-30/qdap"
    expected = read_json(root / "tests/fixtures/qdap-original-sha256.json")
    for relative, digest in expected.items():
        path = root / relative
        # A few local evidence supplements are ignored by Git and absent in a fresh clone.
        if path.exists():
            assert sha(path) == digest
    for name, methods in (("per-query.jsonl", 8), ("main-baseline-per-query.jsonl", 3)):
        records = [
            json.loads(line) for line in (directory / name).read_text().splitlines()
        ]
        counts = Counter(row["method"] for row in records)
        assert len(counts) == methods
        assert set(counts.values()) == {721}
        assert len({row["query_id"] for row in records}) == 721


def test_changed_protocol_is_rejected(portable_run):
    (portable_run.output / "protocol.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="SHA-256 mismatch: protocol"):
        resolve_selection(portable_run)


def test_stage_claim_blocks_overwrites_and_partial_retries(tmp_path):
    from evaluation.qdap_artifacts import claim_stage

    claim_stage(tmp_path, "evaluate")
    with pytest.raises(FileExistsError, match="already started"):
        claim_stage(tmp_path, "evaluate")


def test_prepared_input_content_and_manifest_are_pinned(tmp_path):
    from evaluation.qdap_artifacts import verify_prepared

    (tmp_path / "train.npz").write_bytes(b"prepared")
    write_json(
        tmp_path / "protocol.json",
        {
            "schema_version": 2,
            "prepared_sha256": {"train.npz": sha(tmp_path / "train.npz")},
        },
    )
    write_json(
        tmp_path / "manifest.json",
        {"files": {"protocol.json": sha(tmp_path / "protocol.json")}},
    )
    verify_prepared(tmp_path)
    (tmp_path / "train.npz").write_bytes(b"different")
    with pytest.raises(ValueError, match="train.npz"):
        verify_prepared(tmp_path)

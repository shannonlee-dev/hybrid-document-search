"""QDAP content identities and stage-local bindings (no inference dependencies)."""

import hashlib
import json
import platform
import re
import subprocess
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from evaluation.json_io import read_json

DATA = Path("data/processed/bccard")
VECTORS = Path("artifacts/bccard-adaptive-vectors-resumed")
ROUTER = Path("docs/experiments/adaptive/router.json")


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def check(path, digest, identity):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {identity}: {path}; supply the original artifact"
        )
    if sha(path) != digest:
        raise ValueError(f"SHA-256 mismatch: {identity} ({path})")
    return path


def vector_cache(args):
    return getattr(args, "vector_cache", None) or args.inputs / VECTORS


def model_files(model):
    model = Path(model)
    weights = [*model.glob("*.safetensors"), *model.glob("pytorch_model*.bin")]
    if not weights:
        raise ValueError("local model snapshot contains no recognized weight files")
    return {
        f"model/{p.relative_to(model).as_posix()}": p
        for p in sorted(
            set(
                weights
                + list(model.rglob("*.json"))
                + list(model.rglob("*.model"))
                + list(model.rglob("*.txt"))
            )
        )
    }


def evaluation_files(args):
    data, cache = args.inputs / DATA, vector_cache(args)
    return {
        **{
            f"data/{name}": data / name
            for name in (
                "manifest.json",
                "corpus.jsonl",
                "queries_test.jsonl",
                "qrels_test.jsonl",
            )
        },
        **{
            f"vectors/{name}": cache / name
            for name in (
                "encoding.json",
                "document-vectors.npy",
                "query-vectors-test.npy",
            )
        },
        "router/router.json": args.inputs / ROUTER,
        **model_files(args.model),
    }


def hashes(files):
    return {name: sha(path) for name, path in files.items()}


def model_identity(args):
    revision = getattr(args, "model_revision", None)
    name = getattr(args, "model_name", None)
    if not revision or not name:
        raise ValueError("--model-name and pinned --model-revision are required")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("model revision must be a pinned 40-character commit SHA")
    return {
        "name": name,
        "revision": revision,
        "sha256": hashes(model_files(args.model)),
    }


def verify_model(args, identity):
    for field in ("name", "revision"):
        supplied = getattr(args, f"model_{field}", None)
        if supplied is not None and supplied != identity[field]:
            raise ValueError(f"model {field} mismatch")
    files = model_files(args.model)
    if set(files) != set(identity["sha256"]):
        raise ValueError("model snapshot file set mismatch")
    for key, digest in identity["sha256"].items():
        check(files[key], digest, key)


def provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    packages = {}
    for name in ("numpy", "torch", "transformers", "bm25s", "sentence-transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def legacy_id(source, selection):
    """Translate only known input roots; unrelated historical inputs are not used."""
    path = Path(source)
    for marker, prefix in ((DATA, "data"), (VECTORS, "vectors"), (ROUTER, "router")):
        parts = marker.parts
        for i in range(len(path.parts) - len(parts) + 1):
            if path.parts[i : i + len(parts)] == parts:
                tail = path.parts[i + len(parts) :]
                return f"{prefix}/{'/'.join(tail) if tail else path.name}"
    try:
        return "model/" + path.relative_to(selection["model"]).as_posix()
    except (ValueError, KeyError):
        return None


def selection_metadata(out):
    """Read immutable metadata without requiring checkpoints or external inputs."""
    selection = read_json(out / "selection.json")
    if selection["selection_split"] != "validation":
        raise ValueError("Validation selection required")
    check(out / "protocol.json", selection["protocol_sha256"], "protocol")
    return selection


def resolve_selection(args):
    selection = deepcopy(selection_metadata(args.output))
    files = evaluation_files(args)
    recorded = selection["evaluation_input_sha256"]
    if selection.get("schema_version") != 2:
        translated = {}
        protocol = read_json(args.output / "protocol.json")
        for key, digest in [
            *protocol.get("input_sha256", {}).items(),
            *recorded.items(),
        ]:
            identity = legacy_id(key, selection)
            if identity not in files:
                continue
            if identity in translated and translated[identity] != digest:
                raise ValueError(f"ambiguous legacy identity: {identity}")
            translated[identity] = digest
        recorded = translated
    for identity, path in files.items():
        if identity not in recorded:
            raise ValueError(f"missing recorded evaluation input: {identity}")
        check(path, recorded[identity], identity)
    # A saved model file cannot silently disappear from the supplied snapshot.
    if {k for k in recorded if k and k.startswith("model/")} != {
        k for k in files if k.startswith("model/")
    }:
        raise ValueError("model snapshot file set mismatch")
    if "model_identity" in selection:
        verify_model(args, selection["model_identity"])
    elif getattr(args, "model_revision", None) is not None:
        if args.model_revision != Path(selection["model"]).name:
            raise ValueError("model revision mismatch with legacy snapshot")
    for variant, config in selection["selected"].items():
        path = getattr(args, f"checkpoint_{variant.lower()}", None)
        if path is None:
            saved = Path(config["checkpoint"])
            if saved.is_absolute():
                # Legacy runs retain their candidate subdirectory; never search by basename.
                saved = Path(saved.parent.name) / saved.name
            path = args.output / saved
        config["checkpoint"] = str(
            check(path, config["checkpoint_sha256"], f"{variant} checkpoint")
        )
    selection["model"] = str(args.model)
    return selection


def unused_target(path):
    if Path(path).exists():
        raise FileExistsError(
            f"output already exists: {path}; choose a new run/output directory"
        )


def rows(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def primary(query):
    return query["provenance"] in ("base", "faq_direct")


METHODS = (
    "BM25",
    "Dense",
    "Existing Fixed",
    "Existing Router",
    "Paper Fixed",
    "QDAP-S",
    "QDAP-L",
    "Oracle alpha",
)


def verify_prepared(out, names=None):
    protocol = read_json(out / "protocol.json")
    if protocol.get("schema_version") == 2:
        manifest = read_json(out / "manifest.json")
        for name, digest in manifest["files"].items():
            check(out / name, digest, name)
    for name in names if names is not None else protocol["prepared_sha256"]:
        check(out / name, protocol["prepared_sha256"][name], name)
    return protocol


def claim_stage(out, stage):
    """An exclusive marker also prevents concurrent and partially failed overwrites."""
    marker = out / f".{stage}-started"
    try:
        with marker.open("x") as stream:
            stream.write(
                "Stage started; keep this run as evidence, use a new directory to retry.\n"
            )
    except FileExistsError as error:
        raise FileExistsError(
            f"{stage} already started; choose a new run directory"
        ) from error

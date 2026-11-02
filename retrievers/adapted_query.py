"""Query-only encoder that applies a trained final-block checkpoint to a local model.

Documents stay encoded by the original model; this embedder refuses document
encoding so an adapted model cannot silently re-encode the corpus. Encoding uses
the same DenseEmbedder path (FP32 SentenceTransformer.encode, normalization,
prefix handling) as the original service embedder.

Identity is fail-closed. A checkpoint identity file names the checkpoint SHA-256,
the trained layer count and the base model (name, revision and per-file SHA-256
of the snapshot used in training). Construction refuses to proceed unless the
checkpoint bytes, the local base snapshot files and the saved index's embedding
model all match it.
"""

import hashlib
import json
from pathlib import Path

from retrievers.dense import DenseConfig, DenseEmbedder

IDENTITY_SCHEMA_VERSION = 1
# Same selection rule as the training record (evaluation.query_encoder_report).
_SNAPSHOT_SUFFIXES = (".bin", ".safetensors", ".json", ".model")


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def snapshot_files_sha256(directory: str | Path) -> dict[str, str]:
    """Hash weights, tokenizer, pooling and config files by relative path."""
    directory = Path(directory)
    return {
        str(path.relative_to(directory)): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix in _SNAPSHOT_SUFFIXES
    }


def load_checkpoint_identity(path: str | Path) -> dict:
    """Read and validate a checkpoint identity file; never infer missing fields."""
    identity = json.loads(Path(path).read_text(encoding="utf-8"))
    base = identity.get("base_model") if isinstance(identity, dict) else None
    if (
        not isinstance(identity, dict)
        or identity.get("schema_version") != IDENTITY_SCHEMA_VERSION
        or not _is_digest(identity.get("checkpoint_sha256"))
        or type(identity.get("layers")) is not int
        or identity["layers"] < 1
        or not isinstance(base, dict)
        or not isinstance(base.get("model_name"), str)
        or not isinstance(base.get("revision"), str)
        or not isinstance(base.get("files_sha256"), dict)
        or not base["files_sha256"]
        or not all(_is_digest(v) for v in base["files_sha256"].values())
        or not any(n.endswith((".bin", ".safetensors")) for n in base["files_sha256"])
    ):
        raise ValueError("invalid query checkpoint identity file")
    return identity


def _is_digest(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


class AdaptedQueryEmbedder(DenseEmbedder):
    """Load a verified local snapshot and replace its last N transformer blocks."""

    def __init__(
        self,
        config: DenseConfig,
        *,
        base_model_path: str | Path,
        checkpoint_path: str | Path,
        identity: dict,
    ):
        super().__init__(config)
        base = identity["base_model"]
        self.base_model_path = Path(base_model_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.layers = identity["layers"]
        if not self.base_model_path.is_dir():
            raise FileNotFoundError(f"base model not found: {self.base_model_path}")
        if file_sha256(self.checkpoint_path) != identity["checkpoint_sha256"]:
            raise ValueError("query checkpoint SHA-256 does not match its identity")
        # The saved index records which model encoded the documents; only its
        # name and revision strings are available, so they must match exactly.
        if (config.model_name, config.revision) != (
            base["model_name"],
            base["revision"],
        ):
            raise ValueError("index embedding model differs from the checkpoint base")
        if snapshot_files_sha256(self.base_model_path) != base["files_sha256"]:
            raise ValueError("local base snapshot differs from the checkpoint base")
        self.identity = identity

    def encode_documents(self, texts):
        raise ValueError("adapted query encoder must not encode documents")

    def load_model(self):
        """Load the verified base snapshot and checkpoint once; return the model."""
        if self.model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                str(self.base_model_path),
                device=self.config.device,
                local_files_only=True,
            )
            blocks = model[0].auto_model.encoder.layer
            if self.layers > len(blocks):
                raise ValueError("layers exceeds the model's transformer blocks")
            # Training saved ModuleList state for N > 1 and the bare block for N == 1.
            selected = blocks[-self.layers :] if self.layers > 1 else blocks[-1]
            state = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=True
            )
            selected.load_state_dict(state, strict=True)
            model.eval()
            self.model = model
        return self.model

    def _encode(self, texts, prefix):
        self.load_model()
        return super()._encode(texts, prefix)

"""Query-only encoder that applies a trained final-block checkpoint to a local model.

Documents stay encoded by the original model; this embedder refuses document
encoding so an adapted model cannot silently re-encode the corpus. Encoding uses
the same DenseEmbedder path (FP32 SentenceTransformer.encode, normalization,
prefix handling) as the original service embedder.
"""

import hashlib
from pathlib import Path

from retrievers.dense import DenseConfig, DenseEmbedder


class AdaptedQueryEmbedder(DenseEmbedder):
    """Load a local SentenceTransformer snapshot and replace its last N blocks."""

    def __init__(
        self,
        config: DenseConfig,
        *,
        base_model_path: str | Path,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        layers: int = 4,
    ):
        super().__init__(config)
        if type(layers) is not int or layers < 1:
            raise ValueError("layers must be a positive integer")
        if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
            raise ValueError("checkpoint_sha256 must be a 64-character hex digest")
        self.base_model_path = Path(base_model_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_sha256 = checkpoint_sha256
        self.layers = layers
        if not self.base_model_path.is_dir():
            raise FileNotFoundError(f"base model not found: {self.base_model_path}")
        with self.checkpoint_path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != checkpoint_sha256:
            raise ValueError("query checkpoint SHA-256 does not match")

    def encode_documents(self, texts):
        raise ValueError("adapted query encoder must not encode documents")

    def load_model(self):
        """Load the base snapshot and checkpoint once; return the model."""
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

"""QDAP-S on frozen query vectors, with explicit paper/reference loss variants.

PyTorch is optional for the application; importing this experiment module needs
the existing Dense environment. No encoder or production retriever is modified.
"""

from copy import deepcopy
from typing import Literal

import torch
from torch import Tensor, nn

ALPHA_BINS = 101
CE_WEIGHT = 0.62
WASSERSTEIN_WEIGHT = 0.38
LossMode = Literal["literal", "reference"]
Readout = Literal["mean", "argmax"]


class QDAPSmall(nn.Module):
    """Predict 101 alpha probabilities from detached, frozen query embeddings."""

    def __init__(self, embedding_dim: int = 1024):
        super().__init__()
        if isinstance(embedding_dim, bool) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.linear = nn.Linear(embedding_dim, ALPHA_BINS)
        self.smooth = nn.Conv1d(1, 1, kernel_size=7, padding=3)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Return a distribution; gradients never enter the query encoder."""
        if embeddings.ndim != 2 or embeddings.shape[1] != self.linear.in_features:
            raise ValueError("embeddings must be a batch matching embedding_dim")
        logits = self.linear(embeddings.detach()).unsqueeze(1)
        return self.smooth(logits).squeeze(1).softmax(dim=-1)


class QDAPLarge(nn.Module):
    """Train an independent query encoder and the same 101-bin prediction head.

    The supplied encoder is owned by this predictor. Use from_dense_encoder to
    copy an in-memory retriever; loading a fresh local snapshot also keeps its
    weights independent. No retrieval encoder or corpus embeddings are updated.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)
        self.linear = nn.Linear(encoder.config.hidden_size, ALPHA_BINS)
        self.smooth = nn.Conv1d(1, 1, kernel_size=7, padding=3)

    @classmethod
    def from_dense_encoder(cls, encoder):
        """Deep-copy a retriever's backbone, never sharing parameter storage."""
        return cls(deepcopy(encoder))

    def forward(self, **tokens):
        hidden = self.encoder(**tokens).last_hidden_state[:, 0].float()
        hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
        # The large predictor must keep gradients through the encoder.
        logits = self.smooth(self.linear(hidden).unsqueeze(1)).squeeze(1)
        return logits.float().softmax(-1)


def distribution_targets(utilities: Tensor) -> Tensor:
    """Softmax the 101 graded nDCG utilities without temperature sharpening."""
    utilities = torch.as_tensor(utilities, dtype=torch.float32)
    if utilities.ndim != 2 or utilities.shape[1] != ALPHA_BINS:
        raise ValueError("utilities must have shape (queries, 101)")
    if not torch.isfinite(utilities).all():
        raise ValueError("utilities must be finite")
    return utilities.softmax(dim=-1)


def qdap_loss(
    prediction: Tensor, target: Tensor, *, ce_mode: LossMode = "literal"
) -> Tensor:
    """Combine squared-target CE and summed CDF Wasserstein with .62/.38.

    Literal Eq.10 leaves squared targets unnormalized. The reference variant
    normalizes only the squared CE target; Wasserstein always uses the original
    target. The CDF sum is not divided by the number or spacing of alpha bins.
    """
    if ce_mode not in ("literal", "reference"):
        raise ValueError("ce_mode must be literal or reference")
    _validate_distribution(prediction)
    _validate_distribution(target)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have matching shapes")
    squared = target.square()
    if ce_mode == "reference":
        squared = squared / squared.sum(-1, keepdim=True).clamp(min=1e-8)
    ce = -(squared * (prediction + 1e-8).log()).sum(-1).mean()
    distance = (prediction.cumsum(-1) - target.cumsum(-1)).abs().sum(-1).mean()
    return CE_WEIGHT * ce + WASSERSTEIN_WEIGHT * distance


def read_alpha(probability: Tensor, readout: Readout = "mean") -> Tensor:
    """Read the expectation or the first maximum bin, without rounding."""
    _validate_distribution(probability)
    if readout == "argmax":
        return probability.argmax(-1) / (ALPHA_BINS - 1)
    if readout != "mean":
        raise ValueError("readout must be mean or argmax")
    bins = torch.linspace(0, 1, ALPHA_BINS, device=probability.device)
    return (probability * bins).sum(-1)


def _validate_distribution(value: Tensor) -> None:
    if value.ndim != 2 or value.shape[1] != ALPHA_BINS or not len(value):
        raise ValueError("distribution must have nonempty shape (queries, 101)")
    if (
        not torch.isfinite(value).all()
        or (value < 0).any()
        or not torch.allclose(value.sum(-1), torch.ones_like(value.sum(-1)), atol=1e-5)
    ):
        raise ValueError("distribution must contain finite nonnegative probabilities")

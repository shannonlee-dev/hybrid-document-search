"""QDAP candidate pools, frozen predictions, and evaluation metrics."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fusion.qdap import ALPHA_BINS, QDAPSmall

ALPHAS = np.linspace(0.0, 1.0, ALPHA_BINS)


@dataclass(frozen=True)
class PoolBatch:
    """Label-free, ID-ordered candidate rows and two normalized score channels.

    Invalid padding uses document index -1 and is excluded from all rankings.
    Equal fused scores retain input pool order, which must use document-ID order
    to replay the archived BC Card protocol. No judgments are used at inference.
    """

    indices: np.ndarray
    pairs: np.ndarray
    valid: np.ndarray

    def __post_init__(self):
        if (
            self.indices.ndim != 2
            or not len(self.indices)
            or self.pairs.shape != (*self.indices.shape, 2)
            or self.valid.shape != self.indices.shape
            or self.valid.dtype != bool
            or not np.issubdtype(self.indices.dtype, np.integer)
            or not np.isfinite(self.pairs).all()
        ):
            raise ValueError("invalid pool shapes, indices or scores")
        if (self.indices[self.valid] < 0).any() or (
            self.indices[~self.valid] != -1
        ).any():
            raise ValueError("valid indices must be nonnegative; padding must be -1")
        if not self.valid.any(1).all():
            raise ValueError("each pool needs at least one candidate")
        for row, mask in zip(self.indices, self.valid, strict=True):
            if len(np.unique(row[mask])) != mask.sum():
                raise ValueError("pool candidate indices must be unique")

    @classmethod
    def from_scores(cls, dense, sparse, document_ids, *, depth=100):
        """Union each channel's top-depth, then minmax each over that union."""
        dense, sparse = np.asarray(dense, float), np.asarray(sparse, float)
        if (
            dense.ndim != 2
            or dense.shape != sparse.shape
            or dense.shape[1] != len(document_ids)
            or not len(document_ids)
            or len(set(document_ids)) != len(document_ids)
            or not np.isfinite(dense).all()
            or not np.isfinite(sparse).all()
            or isinstance(depth, bool)
            or not isinstance(depth, int)
            or depth <= 0
        ):
            raise ValueError(
                "finite aligned score matrices and positive depth required"
            )
        width = min(2 * depth, dense.shape[1])
        indices = np.full((len(dense), width), -1, dtype=np.int64)
        pairs = np.zeros((len(dense), width, 2), dtype=np.float64)
        valid = np.zeros(indices.shape, dtype=bool)
        ids = np.asarray(document_ids)
        # 후보 경계 동점은 기존 실험의 corpus 순서를, 최종 동점은 문서 ID를 따른다.
        for i, (d, s) in enumerate(zip(dense, sparse, strict=True)):
            candidates = np.union1d(
                np.argsort(-d, kind="stable")[:depth],
                np.argsort(-s, kind="stable")[:depth],
            )
            candidates = candidates[np.argsort(ids[candidates], kind="stable")]
            size = len(candidates)
            indices[i, :size] = candidates
            valid[i, :size] = True
            for channel, scores in enumerate((d, s)):
                values = scores[candidates]
                spread = np.ptp(values)
                pairs[i, :size, channel] = (
                    (values - values.min()) / spread if spread > 0 else 0
                )
        return cls(indices, pairs, valid)

    @classmethod
    def load(cls, path: Path):
        """Load only inference fields from a pool NPZ, ignoring any labels."""
        with np.load(path, allow_pickle=False) as data:
            return cls(data["pools"], data["pairs"], data["valid"])


def rank_pools(pools: PoolBatch, alpha, *, top_k=10):
    """Rank the fixed pool with dense alpha and stable input-order ties."""
    alpha = np.asarray(alpha, dtype=np.float64)
    if (
        alpha.shape != (len(pools.indices),)
        or not np.isfinite(alpha).all()
        or ((alpha < 0) | (alpha > 1)).any()
    ):
        raise ValueError("alpha must contain one finite weight in [0,1] per query")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    scores = (
        alpha[:, None] * pools.pairs[:, :, 0]
        + (1 - alpha[:, None]) * pools.pairs[:, :, 1]
    )
    scores = np.where(pools.valid, scores, -np.inf)
    order = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    return np.take_along_axis(pools.indices, order, axis=1)


def ranking_metrics(rankings, judgments):
    """Evaluate full qrels using exponential graded gains and binary P/R/MRR.

    Judgments map corpus indices to nonnegative relevance. IDCG includes relevant
    documents outside the candidate pool; -1 padding never counts as a document.
    """
    rankings = np.asarray(rankings)
    if rankings.ndim != 2 or not len(rankings) or len(rankings) != len(judgments):
        raise ValueError("rankings and judgments must align and be nonempty")
    values = []
    for ranking, qrels in zip(rankings, judgments, strict=True):
        if any(
            index < 0 or not math.isfinite(grade) or grade < 0
            for index, grade in qrels.items()
        ):
            raise ValueError("judgments need nonnegative indices and finite grades")
        grades = np.array(
            [qrels.get(int(index), 0) if index >= 0 else 0 for index in ranking]
        )
        relevant = grades > 0
        ideal = sum(
            (2**grade - 1) / math.log2(i + 2)
            for i, grade in enumerate(
                sorted(qrels.values(), reverse=True)[: len(ranking)]
            )
        )
        dcg = ((2**grades - 1) / np.log2(np.arange(len(ranking)) + 2)).sum()
        values.append(
            [
                dcg / ideal if ideal else 0,
                float(relevant[0]) if len(relevant) else 0,
                relevant.sum() / sum(grade > 0 for grade in qrels.values())
                if any(grade > 0 for grade in qrels.values())
                else 0,
                1 / (int(relevant.argmax()) + 1) if relevant.any() else 0,
            ]
        )
    values = np.asarray(values)
    names = ("ndcg@10", "p@1", "recall@10", "mrr@10")
    return dict(zip(names, values.mean(0).tolist(), strict=True)), values


def predict_probabilities(checkpoint, vectors, *, device="cpu", batch_size=256):
    """Load a weights-only QDAP-S head and predict frozen vectors in batches."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("vectors must be a nonempty finite matrix")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model = QDAPSmall(vectors.shape[1]).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    with torch.inference_mode():
        return np.concatenate(
            [
                model(
                    torch.as_tensor(vectors[start : start + batch_size], device=device)
                )
                .cpu()
                .numpy()
                for start in range(0, len(vectors), batch_size)
            ]
        )


def probability_alphas(probability, readout):
    """Use float64 expectation or the first maximum bin, as archived selection did."""
    probability = np.asarray(probability)
    if (
        probability.ndim != 2
        or probability.shape[1] != ALPHA_BINS
        or not np.isfinite(probability).all()
        or (probability < 0).any()
        or not np.allclose(probability.sum(-1), 1, atol=1e-5)
    ):
        raise ValueError("invalid probability distribution")
    if readout == "mean":
        return probability @ ALPHAS
    if readout == "argmax":
        return probability.argmax(1) / (ALPHA_BINS - 1)
    raise ValueError("readout must be mean or argmax")

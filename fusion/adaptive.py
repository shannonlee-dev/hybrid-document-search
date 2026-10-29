"""Three-mode routing over the existing full-corpus Score Fusion contract.

Portable classifier inference is adapted from temp@87cf03e adaptive_v3.py.
No qrels are accepted by the retrieval or prediction APIs.
"""

from time import perf_counter

import numpy as np

from evaluation.json_io import read_json, write_json
from fusion.rrf import validate_top_k
from fusion.score import ScoreFusionRetriever, _fuse_scores, _validate_weight
from indexing.vectors import as_vectors
from retrievers.base import SNIPPET_LENGTH, SearchResult

MODES = ("keyword-heavy", "balanced", "semantic-heavy")
FEATURES = (
    "length",
    "has_digit",
    "digit_ratio",
    "hangul_ratio",
    "latin_ratio",
    "bm25_top",
    "bm25_gap",
    "dense_top",
    "dense_gap",
    "overlap_at_10",
)


def _arrays(sparse, dense, ids):
    sparse, dense = np.asarray(sparse, dtype=float), np.asarray(dense, dtype=float)
    if sparse.shape != (len(ids),) or dense.shape != sparse.shape:
        raise ValueError("scores must match corpus shape")
    if not np.isfinite(sparse).all() or not np.isfinite(dense).all():
        raise ValueError("scores must be finite")
    return sparse, dense


def query_features(query, sparse, dense, ids):
    """Surface features plus deterministic top-10 overlap; see ADAPTIVE_HYBRID.md."""
    sparse, dense = _arrays(sparse, dense, ids)
    text = query.strip()
    n = len(text)
    digit = sum(c.isdigit() for c in text)
    features = [
        float(n),
        float(digit > 0),
        digit / max(n, 1),
        sum("가" <= c <= "힣" for c in text) / max(n, 1),
        sum("a" <= c.lower() <= "z" for c in text) / max(n, 1),
    ]
    orders = []
    for scores, positive_only in ((sparse, True), (dense, False)):
        order = np.lexsort((ids, -scores))
        if positive_only:
            order = order[scores[order] > 0]
        top = scores[order[:2]]
        features.extend(
            [
                float(top[0]) if len(top) else 0.0,
                float(top[0] - top[1]) if len(top) > 1 else 0.0,
            ]
        )
        orders.append(set(order[:10]))
    features.append(len(orders[0] & orders[1]) / min(10, len(ids)) if len(ids) else 0.0)
    return features


def training_labels(quality):
    quality = np.asarray(quality, dtype=float)
    if quality.ndim != 2 or quality.shape[1] != 3 or not len(quality):
        raise ValueError("quality must contain three mode scores per query")
    if not np.isfinite(quality).all():
        raise ValueError("quality must be finite")
    best = quality.max(axis=1)
    tied = np.isclose(quality, best[:, None], rtol=0, atol=1e-12)
    labels = tied.argmax(axis=1)
    labels[tied[:, 1]] = 1
    return labels


def _features(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise ValueError("features must be finite rows in the documented order")
    return x


class Router:
    """Train-only scaler/classifier, serialized as JSON rather than pickle."""

    def __init__(self, state):
        if state.get("version") != 1 or state.get("features") != list(FEATURES):
            raise ValueError("unsupported router schema/features")
        weights = state["weights"]
        if len(weights) != 3:
            raise ValueError("exactly three weights are required")
        for weight in weights:
            _validate_weight(weight)
        if not weights[0] > weights[1] > weights[2] >= 0:
            raise ValueError(
                "weights must descend: keyword-heavy > balanced > semantic-heavy >= 0"
            )
        if state["kind"] not in ("constant", "logistic", "mlp", "tree"):
            raise ValueError("unsupported classifier")
        self.state = state

    @classmethod
    def fit(cls, x, quality, *, weights, kind, split, seed=42):
        if split != "train":
            raise ValueError("classifier fitting requires train split")
        if kind not in ("logistic", "mlp", "tree"):
            raise ValueError("unknown classifier")
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier

        x = _features(x)
        labels = training_labels(quality)
        if len(x) != len(labels):
            raise ValueError("feature/label row counts differ")
        scaler = StandardScaler().fit(x)
        classes = np.unique(labels)
        state = dict(
            version=1,
            features=list(FEATURES),
            weights=list(weights),
            kind=kind,
            seed=seed,
            mean=scaler.mean_.tolist(),
            scale=scaler.scale_.tolist(),
            classes=classes.tolist(),
        )
        if len(classes) == 1:
            state["kind"] = "constant"
            return cls(state)
        if kind == "logistic":
            model = LogisticRegression(C=1, max_iter=2000, random_state=seed)
        elif kind == "mlp":
            model = MLPClassifier(
                hidden_layer_sizes=(16,),
                activation="relu",
                solver="lbfgs",
                alpha=0.0001,
                max_iter=500,
                random_state=seed,
            )
        else:
            model = DecisionTreeClassifier(
                max_depth=3, min_samples_leaf=10, random_state=seed
            )
        model.fit(scaler.transform(x), labels)
        if kind == "logistic":
            state.update(coef=model.coef_.tolist(), intercept=model.intercept_.tolist())
        elif kind == "mlp":
            state["mlp"] = {
                "coefs": [coef.tolist() for coef in model.coefs_],
                "intercepts": [bias.tolist() for bias in model.intercepts_],
                "hidden_layer_sizes": [16],
                "activation": "relu",
                "solver": "lbfgs",
                "alpha": 0.0001,
                "max_iter": 500,
                "iterations": int(model.n_iter_),
                "loss": float(model.loss_),
            }
        else:
            t = model.tree_
            state["tree"] = dict(
                left=t.children_left.tolist(),
                right=t.children_right.tolist(),
                feature=t.feature.tolist(),
                threshold=t.threshold.tolist(),
                value=t.value[:, 0].tolist(),
            )
        return cls(state)

    def select(self, x):
        x = _features(x)
        s = self.state
        classes = np.asarray(s["classes"])
        if s["kind"] == "constant":
            return np.full(len(x), classes[0], dtype=int)
        z = (x - s["mean"]) / s["scale"]
        if s["kind"] == "logistic":
            scores = z @ np.asarray(s["coef"]).T + s["intercept"]
            if len(classes) == 2:
                scores = np.column_stack((np.zeros(len(x)), scores[:, 0]))
        elif s["kind"] == "mlp":
            layers = s["mlp"]
            for weights, bias in zip(
                layers["coefs"][:-1], layers["intercepts"][:-1], strict=True
            ):
                z = np.maximum(0, z @ np.asarray(weights) + bias)
            scores = z @ np.asarray(layers["coefs"][-1]) + layers["intercepts"][-1]
            if len(classes) == 2:
                scores = np.column_stack((np.zeros(len(x)), scores[:, 0]))
        else:
            # sklearn trees compare float32 inputs, including after standardization.
            z = z.astype(np.float32)
            t = s["tree"]
            nodes = np.zeros(len(x), dtype=int)
            for row in range(len(x)):
                node = 0
                while t["left"][node] != -1:
                    side = (
                        "left"
                        if float(z[row, t["feature"][node]]) <= t["threshold"][node]
                        else "right"
                    )
                    node = t[side][node]
                nodes[row] = node
            scores = np.asarray(t["value"])[nodes]
        selected = classes[scores.argmax(axis=1)]
        if 1 in classes:
            default = list(classes).index(1)
            selected[scores[:, default] == scores.max(axis=1)] = 1
        return selected

    def save(self, path):
        write_json(path, self.state)

    @classmethod
    def load(cls, path):
        return cls(read_json(path))


class AdaptiveRetriever(ScoreFusionRetriever):
    """Reuse the validated frozen corpus snapshot; score and encode once per query."""

    def __init__(self, sparse_retriever, dense_retriever, router):
        super().__init__(sparse_retriever, dense_retriever)
        self.router = router
        self._ids = [doc.document_id for doc in self._documents]

    def score_arrays(self, query):
        vector = as_vectors(self._embedder.encode_query(query), self._vectors.shape[1])
        if len(vector) != 1 or not np.isclose(np.linalg.norm(vector[0]), 1, atol=1e-4):
            raise ValueError("requires exactly one normalized query vector")
        dense = (self._vectors @ vector[0]).astype(np.float64)
        return _arrays(self._sparse.score_all(query), dense, self._ids)

    def rank_scores(self, sparse, dense, weight, top_k):
        validate_top_k(top_k)
        _validate_weight(weight)
        sparse, dense = _arrays(sparse, dense, self._ids)
        scores = _fuse_scores(
            dense, sparse, maximum=float(np.max(sparse)), bm25_weight=weight
        )
        selected = np.lexsort((self._tie_ranks, -scores))[:top_k]
        return [
            SearchResult(
                self._ids[row],
                rank,
                float(scores[row]),
                self._documents[row].title,
                self._documents[row].text[:SNIPPET_LENGTH],
            )
            for rank, row in enumerate(selected, 1)
        ]

    def search_details(self, query, top_k):
        validate_top_k(top_k)
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            return [], {
                "mode": "balanced",
                "weight": 0.05,
                "feature_ms": 0.0,
                "router_ms": 0.0,
            }
        sparse, dense = self.score_arrays(query)
        started = perf_counter()
        features = query_features(query, sparse, dense, self._ids)
        feature_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        mode = int(self.router.select([features])[0])
        router_ms = (perf_counter() - started) * 1000
        weight = self.router.state["weights"][mode]
        return self.rank_scores(sparse, dense, weight, top_k), {
            "mode": MODES[mode],
            "weight": weight,
            "feature_ms": feature_ms,
            "router_ms": router_ms,
        }

    def search(self, query, top_k):
        return self.search_details(query, top_k)[0]

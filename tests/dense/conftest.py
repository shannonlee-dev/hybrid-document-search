"""Provide Dense document fixtures and an offline embedding model double."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def dense_corpus_path():
    return Path(__file__).parent / "fixtures" / "dense_corpus.jsonl"


@pytest.fixture
def dense_documents(dense_corpus_path):
    from data.loader import load_prepared_documents

    return load_prepared_documents(dense_corpus_path)


@pytest.fixture
def model_stub(monkeypatch):
    """Install a model double mapping cat, dog and search tokens to orthogonal vectors."""
    import numpy as np

    class _Model:
        def __init__(self, model_name, device=None, revision=None):
            self.model_name = model_name
            self.device = device
            self.revision = revision
            self.inputs = []
            self.options = []

        def _first_module(self):
            return SimpleNamespace(
                auto_model=SimpleNamespace(
                    config=SimpleNamespace(_commit_hash=self.revision)
                )
            )

        def encode(self, texts, **kwargs):
            self.inputs.extend(texts)
            self.options.append(kwargs)
            vectors = np.array(
                [
                    [
                        3 if "고양이" in text else 0,
                        4 if "강아지" in text else 0,
                        5 if "검색" in text else 0,
                    ]
                    for text in texts
                ],
                dtype=np.float32,
            )
            if kwargs["normalize_embeddings"]:
                vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=_Model),
    )
    return _Model

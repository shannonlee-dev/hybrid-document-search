"""Manage embedding settings and document mappings; load models on first encoding."""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from data.loader import Document
from data.preprocess import build_index_text
from indexing.faiss_index import (
    FaissIndex,
    _as_vectors,
    _validate_top_k,
)
from retrievers.base import SearchResult
from retrievers.model_config import COMMIT_SHA_PATTERN, MODELS
from retrievers.model_config import DEFAULT_MODEL as DEFAULT_MODEL
from retrievers.model_config import DEFAULT_MODEL_REVISION as DEFAULT_MODEL_REVISION

if TYPE_CHECKING:
    import numpy as np

DEFAULT_BATCH_SIZE = 32
INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"
_METADATA_VERSION = 1
_SNIPPET_LENGTH = 200
_E5_MODEL_PREFIXES = ("intfloat/e5-", "intfloat/multilingual-e5-")
_E5_INPUT_PREFIXES = (("query_prefix", "query: "), ("passage_prefix", "passage: "))


@dataclass(frozen=True)
class DenseConfig:
    """Pin registered model revisions and apply default prefixes for E5 models."""

    model_name: str = DEFAULT_MODEL
    device: str | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    normalize_embeddings: bool = True
    query_prefix: str | None = None
    passage_prefix: str | None = None
    revision: str | None = None

    def __post_init__(self):
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if self.revision is None:
            object.__setattr__(self, "revision", MODELS.get(self.model_name))
        if self.revision is not None and (
            not isinstance(self.revision, str)
            or not re.fullmatch(COMMIT_SHA_PATTERN, self.revision)
        ):
            raise ValueError("revision must be a fixed 40-character commit SHA or None")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if type(self.normalize_embeddings) is not bool:
            raise ValueError("normalize_embeddings must be a boolean")
        if self.device is not None and (
            not isinstance(self.device, str) or not self.device.strip()
        ):
            raise ValueError("device must be a non-empty string or None")
        is_e5 = self.model_name.startswith(_E5_MODEL_PREFIXES)
        for field, default in _E5_INPUT_PREFIXES:
            prefix = getattr(self, field)
            if prefix is None:
                object.__setattr__(self, field, default if is_e5 else "")
            elif not isinstance(prefix, str):
                raise ValueError(f"{field} must be a string or None")


class DenseEmbedder:
    """Apply shared model and normalization settings to documents and queries."""

    def __init__(self, config: DenseConfig | None = None):
        self.config = config or DenseConfig()
        self.model = None

    def encode_documents(self, texts: list[str]) -> "np.ndarray":
        """Apply the passage prefix and return one FP32 vector per document."""
        if (
            not isinstance(texts, (list, tuple))
            or not texts
            or any(not isinstance(text, str) or not text.strip() for text in texts)
        ):
            raise ValueError("documents must contain non-empty strings")
        return self._encode(texts, self.config.passage_prefix)

    def encode_query(self, query: str) -> "np.ndarray":
        """Apply the query prefix and return an FP32 array with one vector row."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return self._encode([query], self.config.query_prefix)

    def _encode(self, texts: list[str], prefix: str) -> "np.ndarray":
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
                revision=self.config.revision,
            )
        vectors = self.model.encode(
            [prefix + text for text in texts],
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
            # 접두사를 직접 붙였으므로 모델의 기본 프롬프트가 중복 적용되지 않게 한다.
            prompt="",
        )
        vectors = _as_vectors(vectors)
        if len(vectors) != len(texts):
            raise ValueError("model returned an unexpected embedding count")
        return vectors


class DenseRetriever:
    """Preserve FAISS row-to-document mappings and return shared SearchResult records."""

    def __init__(
        self,
        *,
        index: FaissIndex,
        documents: Sequence[Document],
        embedder: DenseEmbedder,
    ) -> None:
        """Copy documents in index row order into an independent mapping snapshot."""
        _validate_documents(documents)
        if index.count != len(documents):
            raise ValueError("index count does not match document mapping")
        snapshot = tuple(
            Document(doc.document_id, doc.text, doc.title) for doc in documents
        )
        self.index = index
        self.documents = snapshot
        self.embedder = embedder
        self.config = embedder.config
        self.build_timings: dict[str, float] | None = None
        self._documents_by_id = {doc.document_id: doc for doc in snapshot}

    @classmethod
    def build(
        cls,
        documents: Sequence[Document],
        config: DenseConfig | None = None,
    ) -> "DenseRetriever":
        """Embed shared title/body text and build an index in input document order."""
        started = perf_counter()
        _validate_documents(documents)
        embedder = DenseEmbedder(config)
        texts = [build_index_text(document) for document in documents]
        embedding_started = perf_counter()
        vectors = embedder.encode_documents(texts)
        embedding_finished = perf_counter()
        index = FaissIndex.build(vectors)
        index_finished = perf_counter()
        retriever = cls(index=index, documents=documents, embedder=embedder)
        retriever.build_timings = {
            "embedding_seconds": embedding_finished - embedding_started,
            "faiss_build_seconds": index_finished - embedding_finished,
            "build_seconds": perf_counter() - started,
        }
        return retriever

    @classmethod
    def load(
        cls, directory: str | Path, *, device: str | None = None
    ) -> "DenseRetriever":
        """Verify checksums, dimensions and document mappings before restoring an index.

        Load the model on first encoding; device=None selects the runtime device
        automatically.
        """
        directory = Path(directory)
        metadata = json.loads(
            (directory / METADATA_FILENAME).read_text(encoding="utf-8")
        )
        if (
            not isinstance(metadata, dict)
            or metadata.get("version") != _METADATA_VERSION
        ):
            raise ValueError("unsupported index metadata version")
        index_path = directory / INDEX_FILENAME
        with index_path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != metadata.get("index_sha256"):
            raise ValueError("index checksum does not match metadata")
        index = FaissIndex.load(index_path)
        if index.dimension != metadata.get("dimension"):
            raise ValueError("saved index dimension does not match metadata")
        if not isinstance(metadata.get("documents"), list) or not metadata["documents"]:
            raise ValueError("invalid saved document mapping metadata")
        try:
            documents = [Document(**doc) for doc in metadata["documents"]]
            config = metadata["embedding_config"]
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid saved index metadata") from exc
        if not isinstance(config, dict):
            raise ValueError("invalid embedding configuration")
        # 이전 저장 형식의 빌드 장치는 무시하고 현재 실행 환경의 device를 적용한다.
        config.pop("device", None)
        # 이전 인덱스의 가중치를 확인할 수 없으므로 현재 후보 SHA를 소급 적용하지 않는다.
        if (
            isinstance(config.get("model_name"), str)
            and config["model_name"] in MODELS
            and config.get("revision") is None
        ):
            raise ValueError(
                "saved registered-model index has no revision; rebuild the index"
            )
        config.setdefault("revision", None)
        if set(config) != {
            field.name for field in fields(DenseConfig) if field.name != "device"
        }:
            raise ValueError("invalid saved embedding configuration")
        try:
            embedder = DenseEmbedder(DenseConfig(**config, device=device))
        except TypeError as exc:
            raise ValueError("invalid saved embedding configuration") from exc
        return cls(index=index, documents=documents, embedder=embedder)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return at most top_k results ranked by inner-product score.

        Empty queries return no results without loading the model. Always validate top_k.
        """
        _validate_top_k(top_k)
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            return []
        hits = self.index.search(self.embedder.encode_query(query), top_k)
        results = []
        for rank, (row, score) in enumerate(hits, start=1):
            doc = self.documents[row]
            results.append(
                SearchResult(
                    document_id=doc.document_id,
                    rank=rank,
                    score=score,
                    title=doc.title,
                    snippet=doc.text[:_SNIPPET_LENGTH],
                )
            )
        return results

    def get_document(self, document_id: str) -> Document:
        return self._documents_by_id[document_id]

    def save(self, directory: str | Path) -> None:
        """Save the index and document mapping with a checksum, omitting the runtime device.

        The two files are not saved atomically; load rejects inconsistent saved pairs.
        """
        directory = Path(directory)
        index_path = directory / INDEX_FILENAME
        self.index.save(index_path)
        with index_path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        config = asdict(self.config)
        config.pop("device")
        metadata = {
            "version": _METADATA_VERSION,
            "dimension": self.index.dimension,
            "index_sha256": digest,
            "embedding_config": config,
            "documents": [
                {"document_id": doc.document_id, "text": doc.text, "title": doc.title}
                for doc in self.documents
            ],
        }
        (directory / METADATA_FILENAME).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _validate_documents(documents: Sequence[Document]) -> None:
    if not isinstance(documents, Sequence) or not documents:
        raise ValueError("corpus must contain documents")
    for doc in documents:
        if not isinstance(doc, Document):
            raise ValueError("corpus must contain shared Document records")
        for name, value in (("document_id", doc.document_id), ("text", doc.text)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if doc.title is not None and not isinstance(doc.title, str):
            raise ValueError("title must be a string or None")
    ids = [doc.document_id for doc in documents]
    if len(set(ids)) != len(ids):
        raise ValueError("document IDs must be unique")

"""임베딩 설정과 문서 매핑을 관리하며 모델은 첫 임베딩 요청에서 로드한다."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

from data.loader import Document
from data.preprocess import build_index_text
from indexing.faiss_index import (
    FaissIndex,
    _as_vectors,
    _validate_top_k,
)
from retrievers.base import SearchResult

if TYPE_CHECKING:
    import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_BATCH_SIZE = 32
INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"
_METADATA_VERSION = 1
_SNIPPET_LENGTH = 200
_E5_MODEL_PREFIXES = ("intfloat/e5-", "intfloat/multilingual-e5-")
_E5_INPUT_PREFIXES = (("query_prefix", "query: "), ("passage_prefix", "passage: "))


@dataclass(frozen=True)
class DenseConfig:
    """E5 계열의 기본 접두사를 적용하고 명시된 접두사는 그대로 보존한다."""

    model_name: str = DEFAULT_MODEL
    device: str | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    normalize_embeddings: bool = True
    query_prefix: str | None = None
    passage_prefix: str | None = None

    def __post_init__(self):
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
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
    """문서와 질의에 동일한 모델과 정규화 설정을 적용한다."""

    def __init__(self, config: DenseConfig | None = None):
        self.config = config or DenseConfig()
        self.model = None

    def encode_documents(self, texts: list[str]) -> "np.ndarray":
        if (
            not isinstance(texts, (list, tuple))
            or not texts
            or any(not isinstance(text, str) or not text.strip() for text in texts)
        ):
            raise ValueError("documents must contain non-empty strings")
        return self._encode(texts, self.config.passage_prefix)

    def encode_query(self, query: str) -> "np.ndarray":
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return self._encode([query], self.config.query_prefix)

    def _encode(self, texts: list[str], prefix: str) -> "np.ndarray":
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.config.model_name, device=self.config.device
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
    """FAISS 행 순서와 문서 매핑을 유지하며 공통 SearchResult 계약을 따른다."""

    def __init__(
        self,
        *,
        index: FaissIndex,
        documents: Sequence[Document],
        embedder: DenseEmbedder,
    ) -> None:
        """완성된 인덱스의 행 순서와 동일한 문서를 받아 독립된 매핑 사본을 보관한다."""
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
        self._documents_by_id = {doc.document_id: doc for doc in snapshot}

    @classmethod
    def build(
        cls,
        documents: Sequence[Document],
        config: DenseConfig | None = None,
    ) -> "DenseRetriever":
        """공통 인덱싱 규칙으로 제목과 본문을 임베딩하고 입력 문서 순서대로 인덱싱한다."""
        _validate_documents(documents)
        embedder = DenseEmbedder(config)
        texts = [build_index_text(document) for document in documents]
        vectors = embedder.encode_documents(texts)
        return cls(
            index=FaissIndex.build(vectors), documents=documents, embedder=embedder
        )

    @classmethod
    def load(
        cls, directory: str | Path, *, device: str | None = None
    ) -> "DenseRetriever":
        """체크섬·차원·문서 매핑을 검증한 뒤 복원하며 device=None이면 장치를 자동 선택한다."""
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
        try:
            documents = [Document(**doc) for doc in metadata["documents"]]
            config = metadata["embedding_config"]
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid saved index metadata") from exc
        if not isinstance(config, dict):
            raise ValueError("invalid embedding configuration")
        # 이전 저장 형식의 빌드 장치는 무시하고 현재 실행 환경의 device를 적용한다.
        config.pop("device", None)
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
        """빈 질의는 모델을 로드하지 않고 빈 결과를 반환하되 top_k 검증은 수행한다."""
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
        """인덱스와 문서 매핑을 체크섬으로 연결해 저장하고 실행 장치는 제외한다.

        두 파일은 원자적으로 저장되지 않으며 불일치한 저장 결과는 load에서 거부한다.
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

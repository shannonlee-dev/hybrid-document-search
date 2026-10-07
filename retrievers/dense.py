"""문서와 질의에 동일한 임베딩 설정을 적용하고 모델을 지연 로드한다."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from indexing.faiss_index import _as_vectors
from retrievers.base import SearchResult

if TYPE_CHECKING:
    import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-base"


DEFAULT_BATCH_SIZE = 32


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
    """Placeholder implementing the shared search signature."""

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return ranked hits once the retrieval implementation is available."""
        raise NotImplementedError("TODO (Role 2): implement DenseRetriever")

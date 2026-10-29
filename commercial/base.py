"""Provider-neutral result and error types for external retrieval services.

External services return their own IDs, optional scores and sometimes several
sub-chunks per original document, so they do not implement `Retriever`.
`evaluation.external` maps their hits back to BC Card IDs before evaluation.
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ProviderHit:
    """One provider-side hit as returned, before mapping to an original ID.

    Fields the provider does not expose stay None; `original_id` is filled only
    from metadata the adapter attached at indexing time, never from text.
    """

    provider_id: str | None
    rank: int | None = None
    score: float | None = None
    original_id: str | None = None
    chunk: str | None = None


@dataclass(frozen=True)
class ProviderResponse:
    hits: tuple[ProviderHit, ...]
    # 원본 응답은 corpus 본문을 포함하므로 Git이 아닌 로컬 artifact에만 기록한다.
    raw: dict = field(default_factory=dict, repr=False)
    usage: dict = field(default_factory=dict)


class ProviderError(Exception):
    """A failed provider request.

    transient errors are retried with bounded backoff; abort errors (auth,
    permission, missing index, exhausted quota) stop the whole run because
    every later request would fail the same way.
    """

    def __init__(
        self,
        message,
        *,
        kind,
        transient=False,
        abort=False,
        retry_after=None,
        status=None,
    ):
        super().__init__(message)
        self.kind = kind
        self.transient = transient
        self.abort = abort
        self.retry_after = retry_after
        self.status = status


class ExternalSearchProvider(Protocol):
    name: str

    def search(self, query: str, top_k: int) -> ProviderResponse: ...

    def describe(self) -> dict:
        """Settings needed to identify the run; must not contain credentials."""
        ...

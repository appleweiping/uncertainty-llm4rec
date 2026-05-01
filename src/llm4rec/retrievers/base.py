"""Retriever contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    user_id: str
    items: list[str]
    scores: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRetriever(Protocol):
    method_name: str

    def fit(
        self,
        train_examples: list[dict[str, Any]],
        item_catalog: list[dict[str, Any]],
        interactions: list[dict[str, Any]] | None = None,
    ) -> None:
        """Fit retriever state."""

    def retrieve(self, example: dict[str, Any], k: int) -> RetrievalResult:
        """Return top-k candidate items."""

"""Unified ranker contract for baseline experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class RankingResult:
    user_id: str
    target_item: str
    candidate_items: list[str]
    predicted_items: list[str]
    scores: list[float]
    method: str
    domain: str
    raw_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prediction_record(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "target_item": self.target_item,
            "candidate_items": self.candidate_items,
            "predicted_items": self.predicted_items,
            "scores": self.scores,
            "method": self.method,
            "domain": self.domain,
            "raw_output": self.raw_output,
            "metadata": self.metadata,
        }


class BaseRanker(Protocol):
    method_name: str

    def fit(
        self,
        train_examples: list[dict[str, Any]],
        item_catalog: list[dict[str, Any]],
        interactions: list[dict[str, Any]] | None = None,
    ) -> None:
        """Fit the ranker using training-only examples."""

    def rank(self, example: dict[str, Any], candidate_items: list[str]) -> RankingResult:
        """Rank candidate item ids for one example."""

    def save(self, path: str | Path) -> None:
        """Optional checkpoint hook."""

    def load(self, path: str | Path) -> None:
        """Optional checkpoint hook."""


class CheckpointNotImplementedMixin:
    def save(self, path: str | Path) -> None:
        raise NotImplementedError("checkpoint save is not implemented for this ranker")

    def load(self, path: str | Path) -> None:
        raise NotImplementedError("checkpoint load is not implemented for this ranker")


def prediction_from_scores(
    *,
    example: dict[str, Any],
    candidate_items: list[str],
    item_scores: dict[str, float],
    method: str,
    metadata: dict[str, Any] | None = None,
) -> RankingResult:
    ordered = sorted(
        [str(item_id) for item_id in candidate_items],
        key=lambda item_id: (-float(item_scores.get(item_id, 0.0)), item_id),
    )
    return RankingResult(
        user_id=str(example["user_id"]),
        target_item=str(example["target"]),
        candidate_items=[str(item_id) for item_id in candidate_items],
        predicted_items=ordered,
        scores=[float(item_scores.get(item_id, 0.0)) for item_id in ordered],
        method=method,
        domain=str(example.get("domain") or "tiny"),
        raw_output=None,
        metadata={
            "example_id": example.get("example_id"),
            "split": example.get("split"),
            **(metadata or {}),
        },
    )

"""Deterministic random ranker baseline."""

from __future__ import annotations

import hashlib
import random
from typing import Any

from llm4rec.rankers.base import CheckpointNotImplementedMixin, RankingResult


class RandomRanker(CheckpointNotImplementedMixin):
    method_name = "random"

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)

    def fit(
        self,
        train_examples: list[dict[str, Any]],
        item_catalog: list[dict[str, Any]],
        interactions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.item_ids = sorted(str(row["item_id"]) for row in item_catalog)

    def rank(self, example: dict[str, Any], candidate_items: list[str]) -> RankingResult:
        ordered = [str(item_id) for item_id in candidate_items]
        rng = random.Random(_stable_seed(self.seed, str(example["example_id"])))
        rng.shuffle(ordered)
        scores = [1.0 / (rank + 1) for rank in range(len(ordered))]
        return RankingResult(
            user_id=str(example["user_id"]),
            target_item=str(example["target"]),
            candidate_items=[str(item_id) for item_id in candidate_items],
            predicted_items=ordered,
            scores=scores,
            method=self.method_name,
            domain=str(example.get("domain") or "tiny"),
            raw_output=None,
            metadata={
                "example_id": example.get("example_id"),
                "split": example.get("split"),
                "seed": self.seed,
                "label_leakage": False,
            },
        )


def _stable_seed(seed: int, example_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)

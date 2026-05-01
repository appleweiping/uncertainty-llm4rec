"""Train-split popularity ranker baseline."""

from __future__ import annotations

from collections import Counter
from typing import Any

from llm4rec.rankers.base import CheckpointNotImplementedMixin, RankingResult, prediction_from_scores


class PopularityRanker(CheckpointNotImplementedMixin):
    method_name = "popularity"

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def fit(
        self,
        train_examples: list[dict[str, Any]],
        item_catalog: list[dict[str, Any]],
        interactions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.counts = Counter()
        for example in train_examples:
            for item_id in example.get("history", []):
                self.counts[str(item_id)] += 1
            self.counts[str(example["target"])] += 1

    def rank(self, example: dict[str, Any], candidate_items: list[str]) -> RankingResult:
        scores = {str(item_id): float(self.counts.get(str(item_id), 0)) for item_id in candidate_items}
        return prediction_from_scores(
            example=example,
            candidate_items=candidate_items,
            item_scores=scores,
            method=self.method_name,
            metadata={
                "score_source": "train_split_item_frequency",
                "unseen_item_default_score": 0.0,
                "label_leakage": False,
            },
        )

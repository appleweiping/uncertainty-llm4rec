"""Thin trainer wrapper for traditional in-memory rankers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.rankers.base import BaseRanker
from llm4rec.trainers.base import TrainResult


class TraditionalRankerTrainer:
    def __init__(
        self,
        ranker: BaseRanker,
        *,
        train_examples: list[dict[str, Any]],
        item_catalog: list[dict[str, Any]],
        interactions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.ranker = ranker
        self.train_examples = train_examples
        self.item_catalog = item_catalog
        self.interactions = interactions

    def train(self) -> TrainResult:
        self.ranker.fit(self.train_examples, self.item_catalog, self.interactions)
        return TrainResult(
            method=self.ranker.method_name,
            artifact_dir=None,
            metadata={
                "train_example_count": len(self.train_examples),
                "item_count": len(self.item_catalog),
                "checkpoint_saved": False,
            },
        )

    def evaluate(self) -> dict[str, Any]:
        return {"method": self.ranker.method_name, "trainer_local_eval": False}

    def predict(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records = []
        for example in examples:
            candidates = [str(item_id) for item_id in example.get("candidates", [])]
            records.append(self.ranker.rank(example, candidates).to_prediction_record())
        return records

    def fit_predict(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.train()
        return self.predict(examples)

    def save_checkpoint(self, path: str | Path) -> None:
        raise NotImplementedError("traditional smoke trainer does not save checkpoints")

    def load_checkpoint(self, path: str | Path) -> None:
        raise NotImplementedError("traditional smoke trainer does not load checkpoints")

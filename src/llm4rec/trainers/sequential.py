"""Sequential trainer for deterministic CPU smoke baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.rankers.base import BaseRanker
from llm4rec.trainers.base import TrainResult
from llm4rec.trainers.checkpointing import load_checkpoint_manifest, save_checkpoint_artifacts


class SequentialTrainer:
    def __init__(
        self,
        ranker: BaseRanker,
        *,
        train_examples: list[dict[str, Any]],
        item_catalog: list[dict[str, Any]],
        interactions: list[dict[str, Any]] | None = None,
        checkpoint_dir: str | Path | None = None,
        config: dict[str, Any] | None = None,
        seed: int = 0,
        eval_only: bool = False,
    ) -> None:
        self.ranker = ranker
        self.train_examples = train_examples
        self.item_catalog = item_catalog
        self.interactions = interactions
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.config = config or {}
        self.seed = int(seed)
        self.eval_only = bool(eval_only)
        self.last_train_result: TrainResult | None = None

    def train(self) -> TrainResult:
        if self.eval_only:
            if self.checkpoint_dir is None:
                raise ValueError("eval-only sequential trainer requires checkpoint_dir")
            self.load_checkpoint(self.checkpoint_dir)
            result = TrainResult(
                method=self.ranker.method_name,
                artifact_dir=str(self.checkpoint_dir),
                checkpoint_dir=str(self.checkpoint_dir),
                metadata={
                    "eval_only": True,
                    "checkpoint_loaded": True,
                    "train_example_count": 0,
                    "item_count": len(self.item_catalog),
                    "seed": self.seed,
                },
            )
            self.last_train_result = result
            return result
        self.ranker.fit(self.train_examples, self.item_catalog, self.interactions)
        checkpoint_metadata: dict[str, Any] = {}
        if self.checkpoint_dir is not None:
            self.save_checkpoint(self.checkpoint_dir)
            checkpoint_metadata = {
                "checkpoint_saved": True,
                "checkpoint_dir": str(self.checkpoint_dir),
            }
        result = TrainResult(
            method=self.ranker.method_name,
            artifact_dir=str(self.checkpoint_dir) if self.checkpoint_dir is not None else None,
            checkpoint_dir=str(self.checkpoint_dir) if self.checkpoint_dir is not None else None,
            metadata={
                "eval_only": False,
                "train_example_count": len(self.train_examples),
                "item_count": len(self.item_catalog),
                "seed": self.seed,
                **checkpoint_metadata,
            },
        )
        self.last_train_result = result
        return result

    def evaluate(self) -> dict[str, Any]:
        return {
            "method": self.ranker.method_name,
            "trainer_local_eval": False,
            "eval_only": self.eval_only,
        }

    def predict(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records = []
        for example in examples:
            candidates = [str(item_id) for item_id in example.get("candidates", [])]
            result = self.ranker.rank(example, candidates)
            record = result.to_prediction_record()
            record["metadata"].update(
                {
                    "trainer": (self.last_train_result.metadata if self.last_train_result else {}),
                    "not_ours_method": True,
                }
            )
            records.append(record)
        return records

    def fit_predict(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.train()
        return self.predict(examples)

    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint_dir = Path(path)
        model_state_path = checkpoint_dir / "model_state.json"
        self.ranker.save(model_state_path)
        save_checkpoint_artifacts(
            checkpoint_dir,
            method=self.ranker.method_name,
            model_state=_read_state(model_state_path),
            config=self.config,
            metadata={
                "trainer": "SequentialTrainer",
                "seed": self.seed,
                "train_example_count": len(self.train_examples),
                "item_count": len(self.item_catalog),
            },
        )

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint_dir = Path(path)
        manifest = load_checkpoint_manifest(checkpoint_dir, expected_method=self.ranker.method_name)
        model_state_path = checkpoint_dir / str(manifest.get("model_state") or "model_state.json")
        self.ranker.load(model_state_path)


def _read_state(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))

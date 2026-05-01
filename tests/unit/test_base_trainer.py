from __future__ import annotations

from pathlib import Path

from llm4rec.rankers.sequential import MarkovSequentialRanker
from llm4rec.trainers.base import TrainResult
from llm4rec.trainers.sequential import SequentialTrainer
from llm4rec.trainers.traditional import TraditionalRankerTrainer
from llm4rec.rankers.popularity import PopularityRanker


ITEMS = [{"item_id": "i1"}, {"item_id": "i2"}]
TRAIN = [{"example_id": "u1:1", "user_id": "u1", "history": ["i1"], "target": "i2", "split": "train"}]
EVAL = [{"example_id": "u2:1", "user_id": "u2", "history": ["i1"], "target": "i2", "candidates": ["i1", "i2"], "split": "test"}]


def test_train_result_has_checkpoint_field() -> None:
    result = TrainResult(method="x", checkpoint_dir="ckpt")
    assert result.checkpoint_dir == "ckpt"


def test_sequential_trainer_train_predict_and_eval_only(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    trainer = SequentialTrainer(
        MarkovSequentialRanker(),
        train_examples=TRAIN,
        item_catalog=ITEMS,
        checkpoint_dir=checkpoint_dir,
        config={"method": {"name": "sequential_markov"}},
        seed=13,
    )
    train_result = trainer.train()
    assert train_result.metadata["checkpoint_saved"] is True
    assert (checkpoint_dir / "checkpoint_manifest.json").exists()
    assert trainer.predict(EVAL)[0]["metadata"]["trainer"]["seed"] == 13

    eval_only = SequentialTrainer(
        MarkovSequentialRanker(),
        train_examples=[],
        item_catalog=ITEMS,
        checkpoint_dir=checkpoint_dir,
        eval_only=True,
    )
    eval_result = eval_only.train()
    assert eval_result.metadata["eval_only"] is True
    assert eval_only.predict(EVAL)[0]["predicted_items"]


def test_traditional_trainer_fit_predict_contract() -> None:
    trainer = TraditionalRankerTrainer(PopularityRanker(), train_examples=TRAIN, item_catalog=ITEMS)
    records = trainer.fit_predict(EVAL)
    assert records[0]["method"] == "popularity"

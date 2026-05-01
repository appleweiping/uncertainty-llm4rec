from __future__ import annotations

from pathlib import Path

from llm4rec.rankers.sequential import MarkovSequentialRanker, SasrecInterfaceRanker, SequentialLastItemRanker


ITEMS = [{"item_id": "i1"}, {"item_id": "i2"}, {"item_id": "i3"}]
TRAIN = [
    {"example_id": "u1:1", "user_id": "u1", "history": ["i1"], "target": "i2", "split": "train"},
    {"example_id": "u2:1", "user_id": "u2", "history": ["i1"], "target": "i2", "split": "train"},
    {"example_id": "u3:1", "user_id": "u3", "history": ["i1"], "target": "i3", "split": "train"},
]
EVAL = {
    "example_id": "u4:1",
    "user_id": "u4",
    "history": ["i1"],
    "target": "i3",
    "candidates": ["i2", "i3"],
    "split": "test",
    "domain": "tiny",
}


def test_markov_sequential_ranker_uses_train_transitions_only() -> None:
    ranker = MarkovSequentialRanker(max_history_length=3)
    ranker.fit(TRAIN, ITEMS)
    result = ranker.rank(EVAL, ["i2", "i3"])
    assert result.predicted_items == ["i2", "i3"]
    assert result.target_item == "i3"
    assert result.metadata["uses_train_transitions"] is True
    assert result.metadata["label_leakage"] is False


def test_markov_checkpoint_roundtrip(tmp_path: Path) -> None:
    ranker = MarkovSequentialRanker(max_history_length=3)
    ranker.fit(TRAIN, ITEMS)
    path = tmp_path / "model_state.json"
    ranker.save(path)
    loaded = MarkovSequentialRanker(max_history_length=1)
    loaded.load(path)
    assert loaded.rank(EVAL, ["i2", "i3"]).predicted_items == ["i2", "i3"]


def test_last_item_and_sasrec_interface_are_explicit_smoke_baselines() -> None:
    last_item = SequentialLastItemRanker()
    last_item.fit(TRAIN, ITEMS)
    last_result = last_item.rank(EVAL, ["i1", "i2"])
    assert last_result.predicted_items[0] == "i1"
    sasrec = SasrecInterfaceRanker()
    sasrec.fit(TRAIN, ITEMS)
    sasrec_result = sasrec.rank(EVAL, ["i2", "i3"])
    assert sasrec_result.method == "sasrec_interface"
    assert sasrec_result.metadata["interface_scaffold"] is True
    assert sasrec_result.metadata["true_sasrec_implemented"] is False

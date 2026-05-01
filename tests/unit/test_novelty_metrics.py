from __future__ import annotations

import math

from llm4rec.metrics.novelty import mean_self_information, novelty_at_k, popularity_ranks, train_item_popularity


def test_train_item_popularity_uses_train_examples_only() -> None:
    popularity = train_item_popularity([
        {"history": ["i1", "i2"], "target": "i1"},
        {"history": ["i2"], "target": "i3"},
    ])
    assert popularity == {"i1": 2, "i2": 2, "i3": 1}


def test_novelty_uses_smoothed_train_popularity() -> None:
    predictions = [{"predicted_items": ["i1", "i3"]}]
    train_popularity = {"i1": 2}
    catalog = ["i1", "i2", "i3"]
    expected = (-math.log2(3 / 5) + -math.log2(1 / 5)) / 2
    assert math.isclose(mean_self_information(predictions, train_popularity=train_popularity, catalog_items=catalog, k=2), expected)
    assert math.isclose(novelty_at_k(predictions, train_popularity=train_popularity, catalog_items=catalog, k=2), expected)


def test_popularity_ranks_are_deterministic() -> None:
    ranks = popularity_ranks({"i2": 2, "i1": 2, "i3": 1}, catalog_items=["i1", "i2", "i3"])
    assert ranks == {"i1": 1, "i2": 2, "i3": 3}

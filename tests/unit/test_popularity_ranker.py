from __future__ import annotations

from llm4rec.rankers.popularity import PopularityRanker


def test_popularity_ranker_counts_train_examples_only() -> None:
    ranker = PopularityRanker()
    train = [
        {"example_id": "u1:1", "user_id": "u1", "history": ["i1"], "target": "i2", "split": "train", "domain": "tiny"},
    ]
    catalog = [{"item_id": "i1"}, {"item_id": "i2"}, {"item_id": "i3"}]
    ranker.fit(train, catalog)
    result = ranker.rank(
        {"example_id": "u2:1", "user_id": "u2", "history": [], "target": "i3", "split": "test", "domain": "tiny"},
        ["i3", "i2", "i1"],
    )
    assert result.predicted_items[:2] == ["i1", "i2"]
    assert result.scores[result.predicted_items.index("i3")] == 0.0
    assert result.method == "popularity"


def test_popularity_ranker_tie_breaks_by_item_id() -> None:
    ranker = PopularityRanker()
    ranker.fit([], [{"item_id": "b"}, {"item_id": "a"}])
    result = ranker.rank(
        {"example_id": "u:1", "user_id": "u", "history": [], "target": "b", "split": "test", "domain": "tiny"},
        ["b", "a"],
    )
    assert result.predicted_items == ["a", "b"]

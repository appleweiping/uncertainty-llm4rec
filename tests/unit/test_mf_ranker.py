from __future__ import annotations

from llm4rec.rankers.mf import MatrixFactorizationRanker


def _train() -> list[dict[str, object]]:
    return [
        {"example_id": "u1:1", "user_id": "u1", "history": ["i1"], "target": "i2", "split": "train", "domain": "tiny"},
        {"example_id": "u2:1", "user_id": "u2", "history": ["i2"], "target": "i3", "split": "train", "domain": "tiny"},
    ]


def _catalog() -> list[dict[str, str]]:
    return [{"item_id": "i1"}, {"item_id": "i2"}, {"item_id": "i3"}]


def test_mf_ranker_is_deterministic_for_same_seed() -> None:
    left = MatrixFactorizationRanker(seed=3, factors=3, epochs=5)
    right = MatrixFactorizationRanker(seed=3, factors=3, epochs=5)
    left.fit(_train(), _catalog())
    right.fit(_train(), _catalog())
    example = {"example_id": "u1:2", "user_id": "u1", "history": ["i1", "i2"], "target": "i3", "split": "test", "domain": "tiny"}
    assert left.rank(example, ["i1", "i2", "i3"]).scores == right.rank(example, ["i1", "i2", "i3"]).scores


def test_mf_ranker_outputs_prediction_schema_shape() -> None:
    ranker = MatrixFactorizationRanker(seed=3, factors=3, epochs=2)
    ranker.fit(_train(), _catalog())
    result = ranker.rank(
        {"example_id": "u1:2", "user_id": "u1", "history": ["i1", "i2"], "target": "i3", "split": "test", "domain": "tiny"},
        ["i1", "i2", "i3"],
    )
    assert result.method == "mf"
    assert len(result.predicted_items) == 3
    assert len(result.scores) == 3
    assert result.metadata["not_bpr"] is True

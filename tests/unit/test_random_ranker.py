from __future__ import annotations

from llm4rec.rankers.random import RandomRanker


def _example(target: str = "i3") -> dict[str, object]:
    return {
        "example_id": "u1:3",
        "user_id": "u1",
        "history": ["i1", "i2"],
        "target": target,
        "split": "test",
        "domain": "tiny",
    }


def test_random_ranker_is_seed_stable() -> None:
    candidates = ["i1", "i2", "i3", "i4"]
    left = RandomRanker(seed=9)
    right = RandomRanker(seed=9)
    left.fit([], [{"item_id": item} for item in candidates])
    right.fit([], [{"item_id": item} for item in candidates])
    assert left.rank(_example(), candidates).predicted_items == right.rank(_example(), candidates).predicted_items


def test_random_ranker_does_not_use_target_for_ordering() -> None:
    candidates = ["i1", "i2", "i3", "i4"]
    ranker = RandomRanker(seed=9)
    ranker.fit([], [{"item_id": item} for item in candidates])
    assert ranker.rank(_example("i3"), candidates).predicted_items == ranker.rank(_example("i4"), candidates).predicted_items

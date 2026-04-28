from __future__ import annotations

import pytest

from storyflow.metrics import assign_popularity_buckets
from storyflow.schemas import PopularityBucket


def test_assign_popularity_buckets_head_mid_tail() -> None:
    buckets = assign_popularity_buckets(
        {
            "item-a": 100,
            "item-b": 80,
            "item-c": 40,
            "item-d": 10,
            "item-e": 1,
        },
        head_fraction=0.2,
        tail_fraction=0.2,
    )

    assert buckets["item-a"] == PopularityBucket.HEAD
    assert buckets["item-c"] == PopularityBucket.MID
    assert buckets["item-e"] == PopularityBucket.TAIL


def test_assign_popularity_buckets_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        assign_popularity_buckets({})
    with pytest.raises(ValueError):
        assign_popularity_buckets({"item": -1})
    with pytest.raises(ValueError):
        assign_popularity_buckets(
            {"item": 1},
            head_fraction=0.8,
            tail_fraction=0.8,
        )

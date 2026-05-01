from __future__ import annotations

from llm4rec.metrics.long_tail import (
    assign_popularity_buckets,
    confidence_by_popularity_bucket,
    coverage_by_popularity_bucket,
    hit_rate_by_popularity_bucket,
)


def test_assign_popularity_buckets_default_fractions() -> None:
    buckets = assign_popularity_buckets(
        {"i1": 10, "i2": 8, "i3": 3, "i4": 1},
        catalog_items=["i1", "i2", "i3", "i4", "i5"],
    )
    assert buckets["i1"] == "head"
    assert buckets["i2"] == "mid"
    assert buckets["i3"] == "mid"
    assert buckets["i5"] == "tail"


def test_long_tail_bucket_metrics() -> None:
    item_buckets = {"i1": "head", "i2": "mid", "i3": "tail"}
    predictions = [
        {"target_item": "i1", "predicted_items": ["i1"], "metadata": {"confidence": 0.9}},
        {"target_item": "i2", "predicted_items": ["i3"], "metadata": {"confidence": 0.5}},
    ]
    assert hit_rate_by_popularity_bucket(predictions, item_buckets=item_buckets, k=1) == {"head": 1.0, "mid": 0.0}
    assert coverage_by_popularity_bucket(predictions, item_buckets=item_buckets, k=1) == {"head": 1.0, "mid": 0.0, "tail": 1.0}
    assert confidence_by_popularity_bucket(predictions, item_buckets=item_buckets)["head"]["mean_confidence"] == 0.9

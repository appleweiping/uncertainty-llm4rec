"""Popularity bucket utilities."""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil

from storyflow.schemas import PopularityBucket


def assign_popularity_buckets(
    popularity_by_item: Mapping[str, float],
    *,
    head_fraction: float = 0.2,
    tail_fraction: float = 0.2,
) -> dict[str, PopularityBucket]:
    """Assign deterministic head/mid/tail buckets from item popularity counts."""

    if not popularity_by_item:
        raise ValueError("popularity_by_item must not be empty")
    if not 0.0 <= head_fraction <= 1.0:
        raise ValueError("head_fraction must be in [0, 1]")
    if not 0.0 <= tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in [0, 1]")
    if head_fraction + tail_fraction > 1.0:
        raise ValueError("head_fraction + tail_fraction must be <= 1")
    for item_id, popularity in popularity_by_item.items():
        if not item_id:
            raise ValueError("item ids must be non-empty strings")
        if popularity < 0:
            raise ValueError("popularity values must be non-negative")

    ranked_items = sorted(
        popularity_by_item.items(),
        key=lambda item: (-item[1], item[0]),
    )
    n_items = len(ranked_items)
    n_head = ceil(n_items * head_fraction) if head_fraction > 0 else 0
    n_tail = ceil(n_items * tail_fraction) if tail_fraction > 0 else 0
    if n_items == 1:
        n_head, n_tail = 1, 0

    buckets: dict[str, PopularityBucket] = {}
    head_ids = {item_id for item_id, _ in ranked_items[:n_head]}
    tail_start = n_items - n_tail
    tail_ids = {
        item_id
        for item_id, _ in ranked_items[max(tail_start, n_head):]
    }
    for item_id, _ in ranked_items:
        if item_id in head_ids:
            buckets[item_id] = PopularityBucket.HEAD
        elif item_id in tail_ids:
            buckets[item_id] = PopularityBucket.TAIL
        else:
            buckets[item_id] = PopularityBucket.MID
    return buckets

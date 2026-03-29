# src/data/popularity.py
from __future__ import annotations

from collections import Counter
from typing import Iterable


def compute_item_popularity(interactions: Iterable[list[str]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for seq in interactions:
        counter.update(seq)
    return dict(counter)


def build_popularity_groups(
    popularity: dict[str, int],
    head_ratio: float = 0.2,
    mid_ratio: float = 0.6,
) -> dict[str, str]:
    """
    按频次排序后分成 head / mid / tail
    """
    items = sorted(popularity.items(), key=lambda x: x[1], reverse=True)
    n = len(items)
    head_end = max(1, int(n * head_ratio))
    mid_end = max(head_end + 1, int(n * mid_ratio))

    groups: dict[str, str] = {}
    for idx, (item_id, _) in enumerate(items):
        if idx < head_end:
            groups[item_id] = "head"
        elif idx < mid_end:
            groups[item_id] = "mid"
        else:
            groups[item_id] = "tail"
    return groups
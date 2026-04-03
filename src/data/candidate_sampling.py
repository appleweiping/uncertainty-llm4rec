# src/data/candidate_sampling.py

from __future__ import annotations

import random
from typing import List, Sequence, Set


def sample_negative_items(
    user_seen_items: Set[str],
    all_item_ids: Sequence[str],
    num_negatives: int,
    rng: random.Random,
) -> List[str]:
    """
    从全量 item 中采样当前用户未交互过的负样本。

    Args:
        user_seen_items: 当前用户已经看过/交互过的 item_id 集合
        all_item_ids: 全量 item_id 列表
        num_negatives: 需要采样的负样本数量
        rng: 随机数生成器，保证可复现

    Returns:
        负样本 item_id 列表
    """
    if num_negatives <= 0:
        return []

    candidates = [item_id for item_id in all_item_ids if item_id not in user_seen_items]

    if not candidates:
        return []

    if len(candidates) <= num_negatives:
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        return shuffled

    return rng.sample(candidates, num_negatives)
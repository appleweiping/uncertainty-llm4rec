# src/data/candidate_sampling.py

from __future__ import annotations

import random
from typing import List, Sequence, Set


def _normalize_item_id_list(item_ids: Sequence[str]) -> List[str]:
    normalized = []
    for x in item_ids:
        if x is None:
            continue
        x = str(x).strip()
        if x == "":
            continue
        normalized.append(x)
    return normalized


def _normalize_item_id_set(item_ids: Set[str]) -> Set[str]:
    normalized = set()
    for x in item_ids:
        if x is None:
            continue
        x = str(x).strip()
        if x == "":
            continue
        normalized.add(x)
    return normalized


def sample_negative_items(
    user_seen_items: Set[str],
    all_item_ids: Sequence[str],
    num_negatives: int,
    rng: random.Random,
) -> List[str]:
    """
    从全量 item 中采样用户未交互过的负样本。

    Args:
        user_seen_items: 当前用户已见过的 item_id 集合
        all_item_ids: 全量 item_id 列表
        num_negatives: 需要采样的负样本数量
        rng: 随机数生成器，保证可复现

    Returns:
        负样本 item_id 列表
    """
    if num_negatives <= 0:
        return []

    seen = _normalize_item_id_set(user_seen_items)
    all_items = _normalize_item_id_list(all_item_ids)

    # 去重同时保序
    dedup_all_items = list(dict.fromkeys(all_items))

    candidates = [item_id for item_id in dedup_all_items if item_id not in seen]

    if len(candidates) == 0:
        return []

    if len(candidates) <= num_negatives:
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        return shuffled

    return rng.sample(candidates, num_negatives)
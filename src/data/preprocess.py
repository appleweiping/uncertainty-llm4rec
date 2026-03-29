# src/data/preprocess.py
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from src.data.candidate_sampling import RandomCandidateSampler


@dataclass
class Item:
    item_id: str
    title: str
    meta: str = ""


@dataclass
class Sample:
    user_id: str
    history_items: list[dict[str, Any]]
    target_item: dict[str, Any]
    candidates: list[dict[str, Any]]
    label: int
    target_popularity_group: str


def _item_dict(item_id: str, item_meta: dict[str, dict[str, str]]) -> dict[str, Any]:
    meta = item_meta.get(item_id, {})
    return asdict(
        Item(
            item_id=item_id,
            title=meta.get("title", f"Item {item_id}"),
            meta=meta.get("meta", ""),
        )
    )


def build_samples(
    interactions: dict[str, list[str]],
    item_meta: dict[str, dict[str, str]],
    popularity_group_map: dict[str, str],
    max_history: int = 20,
    n_neg: int = 19,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    all_item_ids = list(item_meta.keys())
    sampler = RandomCandidateSampler(all_item_ids=all_item_ids, seed=seed)

    samples: list[dict[str, Any]] = []

    for user_id, seq in interactions.items():
        if len(seq) < 3:
            continue

        history = seq[:-1][-max_history:]
        target = seq[-1]
        history_set = set(history)

        neg_items = sampler.sample(user_history=history_set, pos_item=target, n_neg=n_neg)
        candidate_ids = [target] + neg_items
        rng.shuffle(candidate_ids)

        sample = Sample(
            user_id=user_id,
            history_items=[_item_dict(x, item_meta) for x in history],
            target_item=_item_dict(target, item_meta),
            candidates=[_item_dict(x, item_meta) for x in candidate_ids],
            label=1,
            target_popularity_group=popularity_group_map.get(target, "tail"),
        )
        samples.append(asdict(sample))

    return samples
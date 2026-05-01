"""Candidate-set construction with deterministic seed handling."""

from __future__ import annotations

import hashlib
import random
from dataclasses import replace

from llm4rec.data.base import ItemRecord, UserExample


def build_candidate_items(
    example: UserExample,
    items: list[ItemRecord],
    *,
    protocol: str = "full",
    include_history: bool = False,
    sample_size: int | None = None,
    seed: int = 0,
) -> list[str]:
    """Build a deterministic candidate list and always preserve the target."""

    if protocol not in {"full", "sampled"}:
        raise ValueError("candidate protocol must be full or sampled")
    item_ids = sorted({item.item_id for item in items})
    history = set(example.history)
    pool = [
        item_id
        for item_id in item_ids
        if include_history or item_id not in history or item_id == example.target
    ]
    if example.target not in pool:
        pool.append(example.target)
    pool = sorted(set(pool))
    if protocol == "full":
        return pool
    if sample_size is None or sample_size < 1:
        raise ValueError("sampled candidate protocol requires sample_size >= 1")
    negatives = [item_id for item_id in pool if item_id != example.target]
    rng = random.Random(_example_seed(seed, example.example_id))
    rng.shuffle(negatives)
    selected = [example.target] + negatives[: max(0, sample_size - 1)]
    return sorted(set(selected))


def attach_candidates(
    examples: list[UserExample],
    items: list[ItemRecord],
    *,
    protocol: str,
    include_history: bool,
    sample_size: int | None,
    seed: int,
) -> list[UserExample]:
    return [
        replace(
            example,
            candidates=build_candidate_items(
                example,
                items,
                protocol=protocol,
                include_history=include_history,
                sample_size=sample_size,
                seed=seed,
            ),
        )
        for example in examples
    ]


def _example_seed(seed: int, example_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)

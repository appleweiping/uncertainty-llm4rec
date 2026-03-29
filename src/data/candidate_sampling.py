# src/data/candidate_sampling.py
from __future__ import annotations

import random
from typing import Sequence


class RandomCandidateSampler:
    def __init__(self, all_item_ids: Sequence[str], seed: int = 42) -> None:
        self.all_item_ids = list(all_item_ids)
        self.rng = random.Random(seed)

    def sample(
        self,
        user_history: set[str],
        pos_item: str,
        n_neg: int = 19,
    ) -> list[str]:
        pool = [x for x in self.all_item_ids if x != pos_item and x not in user_history]
        if len(pool) < n_neg:
            return self.rng.sample(pool, k=len(pool))
        return self.rng.sample(pool, k=n_neg)
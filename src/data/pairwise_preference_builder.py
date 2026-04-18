from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PairwisePreferenceBuildConfig:
    pair_type: str = "positive_vs_negative"
    shuffle_pair_order: bool = True
    shuffle_seed: int = 42
    max_pairs_per_sample: Optional[int] = None


def _stable_rng(seed: int, *parts: str) -> random.Random:
    digest = hashlib.sha256("||".join([str(seed), *parts]).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _candidate_record(sample: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "item_id": str(sample["candidate_item_ids"][index]).strip(),
        "title": str(sample["candidate_titles"][index]).strip(),
        "text": str(sample["candidate_texts"][index]).strip(),
        "popularity_group": str(sample["candidate_popularity_groups"][index]).strip() or "mid",
        "label": int(sample["candidate_labels"][index]),
    }


def build_pairwise_preferences_from_ranking_samples(
    ranking_samples: List[Dict[str, Any]],
    *,
    cfg: PairwisePreferenceBuildConfig,
) -> List[Dict[str, Any]]:
    pairwise_samples: List[Dict[str, Any]] = []

    for sample in ranking_samples:
        source_event_id = str(sample["source_event_id"])
        positive_index = int(sample["positive_item_index"])
        positive = _candidate_record(sample, positive_index)

        negative_indices = [
            idx
            for idx, label in enumerate(sample["candidate_labels"])
            if int(label) == 0
        ]

        if cfg.max_pairs_per_sample is not None:
            negative_indices = negative_indices[: cfg.max_pairs_per_sample]

        for pair_offset, negative_index in enumerate(negative_indices):
            negative = _candidate_record(sample, negative_index)

            item_a = positive
            item_b = negative

            if cfg.shuffle_pair_order:
                rng = _stable_rng(cfg.shuffle_seed, source_event_id, str(negative["item_id"]), str(pair_offset))
                if rng.random() < 0.5:
                    item_a, item_b = item_b, item_a

            pairwise_samples.append(
                {
                    "pair_id": f"{source_event_id}::pair::{pair_offset}",
                    "source_event_id": source_event_id,
                    "user_id": str(sample["user_id"]),
                    "history": sample.get("history", []),
                    "item_a_id": item_a["item_id"],
                    "item_a_title": item_a["title"],
                    "item_a_text": item_a["text"],
                    "item_a_popularity_group": item_a["popularity_group"],
                    "item_b_id": item_b["item_id"],
                    "item_b_title": item_b["title"],
                    "item_b_text": item_b["text"],
                    "item_b_popularity_group": item_b["popularity_group"],
                    "preferred_item": positive["item_id"],
                    "pair_type": cfg.pair_type,
                    "source_positive_item_id": positive["item_id"],
                    "timestamp": sample["timestamp"],
                    "split_name": sample["split_name"],
                    "source_candidate_count": int(sample["num_candidates"]),
                }
            )

    return pairwise_samples

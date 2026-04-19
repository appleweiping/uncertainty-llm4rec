from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PairwisePreferenceBuildConfig:
    pair_type: str = "positive_vs_negative"
    pair_generation_mode: str = "positive_vs_negative"
    shuffle_pair_order: bool = True
    shuffle_seed: int = 42
    max_pairs_per_sample: Optional[int] = None
    max_pairs_per_event: Optional[int] = None
    event_balanced_order: bool = False


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
    event_pair_groups: List[List[Dict[str, Any]]] = []
    mode = cfg.pair_generation_mode.strip().lower()

    for sample in ranking_samples:
        source_event_id = str(sample["source_event_id"])
        positive_index = int(sample["positive_item_index"])
        positive = _candidate_record(sample, positive_index)

        negative_indices = [
            idx
            for idx, label in enumerate(sample["candidate_labels"])
            if int(label) == 0
        ]

        if mode in {"positive_vs_negative", "positive_vs_all_negatives"}:
            selected_negative_indices = negative_indices
        elif mode in {"event_balanced_positive_vs_negative", "coverage_balanced_positive_vs_negative"}:
            selected_negative_indices = negative_indices
        elif mode == "local_positive_neighbors":
            selected_negative_indices = sorted(
                negative_indices,
                key=lambda idx: (abs(idx - positive_index), idx),
            )
        else:
            raise ValueError(f"Unsupported pair_generation_mode: {cfg.pair_generation_mode}")

        max_pairs = cfg.max_pairs_per_event
        if max_pairs is None:
            max_pairs = cfg.max_pairs_per_sample
        if max_pairs is not None:
            selected_negative_indices = selected_negative_indices[: int(max_pairs)]

        event_pairs: List[Dict[str, Any]] = []
        for pair_offset, negative_index in enumerate(selected_negative_indices):
            negative = _candidate_record(sample, negative_index)

            item_a = positive
            item_b = negative

            if cfg.shuffle_pair_order:
                rng = _stable_rng(cfg.shuffle_seed, source_event_id, str(negative["item_id"]), str(pair_offset))
                if rng.random() < 0.5:
                    item_a, item_b = item_b, item_a

            event_pairs.append(
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
                    "pair_generation_mode": cfg.pair_generation_mode,
                }
            )
        event_pair_groups.append(event_pairs)

    pairwise_samples: List[Dict[str, Any]] = []
    if cfg.event_balanced_order:
        max_group_len = max((len(group) for group in event_pair_groups), default=0)
        for pair_idx in range(max_group_len):
            for group in event_pair_groups:
                if pair_idx < len(group):
                    pairwise_samples.append(group[pair_idx])
    else:
        for group in event_pair_groups:
            pairwise_samples.extend(group)

    if cfg.max_pairs_per_sample is not None and cfg.event_balanced_order:
        pairwise_samples = pairwise_samples[: int(cfg.max_pairs_per_sample)]
    return pairwise_samples

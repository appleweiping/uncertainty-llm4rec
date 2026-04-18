from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class CandidateRankingBuildConfig:
    shuffle_candidates: bool = True
    shuffle_seed: int = 42
    require_single_positive: bool = True


def _event_key(user_id: str, timestamp: Any) -> str:
    return f"{user_id}::{timestamp}"


def _stable_rng(seed: int, *parts: str) -> random.Random:
    digest = hashlib.sha256("||".join([str(seed), *parts]).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def group_pointwise_records_by_event(
    records: Iterable[Dict[str, Any]],
) -> Dict[Tuple[str, Any], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}

    for record in records:
        user_id = str(record.get("user_id", "")).strip()
        timestamp = record.get("timestamp")
        if user_id == "" or timestamp is None:
            raise ValueError("Pointwise record missing user_id or timestamp.")

        key = (user_id, timestamp)
        grouped.setdefault(key, []).append(record)

    return grouped


def _validate_pointwise_group(
    group: List[Dict[str, Any]],
    *,
    cfg: CandidateRankingBuildConfig,
    split_name: str,
) -> None:
    if not group:
        raise ValueError("Empty pointwise group.")

    history = group[0].get("history", [])
    positive_cnt = sum(int(row.get("label", 0)) == 1 for row in group)

    for row in group:
        if row.get("history", []) != history:
            raise ValueError(f"[{split_name}] Inconsistent history found inside one event group.")

    if cfg.require_single_positive and positive_cnt != 1:
        user_id = str(group[0].get("user_id", ""))
        timestamp = group[0].get("timestamp")
        raise ValueError(
            f"[{split_name}] Expected exactly one positive item for {user_id} @ {timestamp}, got {positive_cnt}."
        )


def _shuffle_group_records(
    group: List[Dict[str, Any]],
    *,
    cfg: CandidateRankingBuildConfig,
) -> List[Dict[str, Any]]:
    if not cfg.shuffle_candidates:
        return list(group)

    user_id = str(group[0]["user_id"])
    timestamp = str(group[0]["timestamp"])
    rng = _stable_rng(cfg.shuffle_seed, user_id, timestamp)
    shuffled = list(group)
    rng.shuffle(shuffled)
    return shuffled


def build_candidate_ranking_samples_from_pointwise(
    records: List[Dict[str, Any]],
    *,
    split_name: str,
    cfg: CandidateRankingBuildConfig,
) -> List[Dict[str, Any]]:
    grouped = group_pointwise_records_by_event(records)
    ranking_samples: List[Dict[str, Any]] = []

    for (user_id, timestamp), group in grouped.items():
        _validate_pointwise_group(group, cfg=cfg, split_name=split_name)
        ordered_group = _shuffle_group_records(group, cfg=cfg)

        positive_row = next((row for row in ordered_group if int(row.get("label", 0)) == 1), None)
        if positive_row is None:
            continue

        candidate_item_ids = [str(row.get("candidate_item_id", "")).strip() for row in ordered_group]
        candidate_titles = [str(row.get("candidate_title", "")).strip() for row in ordered_group]
        candidate_texts = [str(row.get("candidate_text", "")).strip() for row in ordered_group]
        candidate_popularity_groups = [
            str(row.get("target_popularity_group", "mid")).strip() or "mid"
            for row in ordered_group
        ]
        candidate_labels = [int(row.get("label", 0)) for row in ordered_group]

        positive_item_id = str(positive_row.get("candidate_item_id", "")).strip()
        positive_index = candidate_item_ids.index(positive_item_id)

        ranking_samples.append(
            {
                "source_event_id": _event_key(user_id, timestamp),
                "user_id": user_id,
                "history": positive_row.get("history", []),
                "candidate_item_ids": candidate_item_ids,
                "candidate_titles": candidate_titles,
                "candidate_texts": candidate_texts,
                "candidate_popularity_groups": candidate_popularity_groups,
                "candidate_labels": candidate_labels,
                "positive_item_id": positive_item_id,
                "positive_item_title": str(positive_row.get("candidate_title", "")).strip(),
                "positive_item_text": str(positive_row.get("candidate_text", "")).strip(),
                "positive_item_index": positive_index,
                "timestamp": timestamp,
                "split_name": split_name,
                "num_candidates": len(candidate_item_ids),
                "source_pointwise_size": len(group),
            }
        )

    return ranking_samples

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.data.candidate_sampling import sample_negative_items


@dataclass
class BuildSamplesConfig:
    max_history_len: int = 10
    num_negatives: int = 5
    seed: int = 42


def _normalize_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_item_lookup(items_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    if "item_id" not in items_df.columns:
        raise ValueError("items.csv 缺少必要字段: item_id")
    if "candidate_text" not in items_df.columns:
        raise ValueError("items.csv 缺少必要字段: candidate_text")

    lookup: Dict[str, Dict[str, str]] = {}

    df = items_df.copy()
    df["item_id"] = df["item_id"].map(_normalize_id)

    for _, row in df.iterrows():
        item_id = row["item_id"]
        if item_id == "":
            continue

        lookup[item_id] = {
            "candidate_title": str(row["title"]).strip()
            if "title" in df.columns and pd.notna(row["title"])
            else "",
            "candidate_text": str(row["candidate_text"]).strip()
            if pd.notna(row["candidate_text"])
            else "",
        }

    return lookup


def build_popularity_lookup(popularity_df: pd.DataFrame) -> Dict[str, str]:
    required_cols = ["item_id", "popularity_group"]
    for col in required_cols:
        if col not in popularity_df.columns:
            raise ValueError(f"popularity_stats.csv 缺少必要字段: {col}")

    lookup: Dict[str, str] = {}

    df = popularity_df.copy()
    df["item_id"] = df["item_id"].map(_normalize_id)

    for _, row in df.iterrows():
        item_id = row["item_id"]
        if item_id == "":
            continue
        group = str(row["popularity_group"]).strip() if pd.notna(row["popularity_group"]) else "mid"
        if group == "":
            group = "mid"
        lookup[item_id] = group

    return lookup


def sort_and_group_interactions(interactions_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    required_cols = ["user_id", "item_id", "timestamp"]
    for col in required_cols:
        if col not in interactions_df.columns:
            raise ValueError(f"interactions.csv 缺少必要字段: {col}")

    df = interactions_df.copy()
    df["user_id"] = df["user_id"].map(_normalize_id)
    df["item_id"] = df["item_id"].map(_normalize_id)

    df = df[(df["user_id"] != "") & (df["item_id"] != "")]
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    user_sequences: Dict[str, List[Dict[str, Any]]] = {}

    for user_id, group in df.groupby("user_id"):
        seq: List[Dict[str, Any]] = []
        for _, row in group.iterrows():
            seq.append(
                {
                    "item_id": row["item_id"],
                    "timestamp": row["timestamp"],
                }
            )
        if len(seq) > 0:
            user_sequences[user_id] = seq

    return user_sequences


def deduplicate_user_sequences(
    user_sequences: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    对每个用户去重 item，只保留该 item 第一次出现。
    目标：把任务定义固定为“推荐未见过的新 item”，而不是复购预测。
    """
    deduped: Dict[str, List[Dict[str, Any]]] = {}

    for user_id, seq in user_sequences.items():
        seen_items = set()
        new_seq: List[Dict[str, Any]] = []

        for event in seq:
            item_id = _normalize_id(event["item_id"])
            if item_id == "":
                continue
            if item_id in seen_items:
                continue
            seen_items.add(item_id)
            new_seq.append(event)

        if len(new_seq) > 0:
            deduped[user_id] = new_seq

    return deduped


def split_user_sequence_leave_one_out(
    user_sequences: Dict[str, List[Dict[str, Any]]]
) -> Tuple[
    Dict[str, List[Dict[str, Any]]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
]:
    train_histories: Dict[str, List[Dict[str, Any]]] = {}
    valid_targets: Dict[str, Dict[str, Any]] = {}
    test_targets: Dict[str, Dict[str, Any]] = {}

    for user_id, seq in user_sequences.items():
        if len(seq) < 3:
            continue

        train_histories[user_id] = seq[:-2]
        valid_targets[user_id] = seq[-2]
        test_targets[user_id] = seq[-1]

    return train_histories, valid_targets, test_targets


def _format_history(
    history_item_ids: List[str],
    item_lookup: Dict[str, Dict[str, str]],
    max_history_len: int,
) -> List[str]:
    trimmed = history_item_ids[-max_history_len:]
    history_texts: List[str] = []

    for item_id in trimmed:
        item_info = item_lookup.get(item_id, {})
        title = item_info.get("candidate_title", "").strip()
        candidate_text = item_info.get("candidate_text", "").strip()

        if title:
            history_texts.append(title)
        elif candidate_text:
            history_texts.append(candidate_text[:200])
        else:
            history_texts.append(item_id)

    return history_texts


def _build_single_record(
    user_id: str,
    history_item_ids: List[str],
    candidate_item_id: str,
    label: int,
    timestamp: Any,
    item_lookup: Dict[str, Dict[str, str]],
    popularity_lookup: Dict[str, str],
    cfg: BuildSamplesConfig,
) -> Dict[str, Any]:
    item_info = item_lookup.get(candidate_item_id, {})

    return {
        "user_id": user_id,
        "history": _format_history(
            history_item_ids=history_item_ids,
            item_lookup=item_lookup,
            max_history_len=cfg.max_history_len,
        ),
        "candidate_item_id": candidate_item_id,
        "candidate_title": item_info.get("candidate_title", ""),
        "candidate_text": item_info.get("candidate_text", ""),
        "label": int(label),
        "target_popularity_group": popularity_lookup.get(candidate_item_id, "mid"),
        "timestamp": timestamp,
    }


def build_train_pointwise_samples(
    train_histories: Dict[str, List[Dict[str, Any]]],
    item_lookup: Dict[str, Dict[str, str]],
    popularity_lookup: Dict[str, str],
    cfg: BuildSamplesConfig,
) -> List[Dict[str, Any]]:
    rng = random.Random(cfg.seed)
    all_item_ids = list(item_lookup.keys())
    records: List[Dict[str, Any]] = []

    for user_id, seq in train_histories.items():
        if len(seq) < 2:
            continue

        full_seen_items = {_normalize_id(x["item_id"]) for x in seq if _normalize_id(x["item_id"]) != ""}

        for idx in range(1, len(seq)):
            history_seq = seq[:idx]
            target = seq[idx]

            history_item_ids = [_normalize_id(x["item_id"]) for x in history_seq if _normalize_id(x["item_id"]) != ""]
            target_item_id = _normalize_id(target["item_id"])
            target_timestamp = target["timestamp"]

            if target_item_id == "" or target_item_id not in item_lookup:
                continue

            if target_item_id in set(history_item_ids):
                continue

            pos_record = _build_single_record(
                user_id=user_id,
                history_item_ids=history_item_ids,
                candidate_item_id=target_item_id,
                label=1,
                timestamp=target_timestamp,
                item_lookup=item_lookup,
                popularity_lookup=popularity_lookup,
                cfg=cfg,
            )
            records.append(pos_record)

            negative_item_ids = sample_negative_items(
                user_seen_items=full_seen_items,
                all_item_ids=all_item_ids,
                num_negatives=cfg.num_negatives,
                rng=rng,
            )

            for neg_item_id in negative_item_ids:
                if neg_item_id not in item_lookup:
                    continue
                if neg_item_id in set(history_item_ids):
                    continue

                neg_record = _build_single_record(
                    user_id=user_id,
                    history_item_ids=history_item_ids,
                    candidate_item_id=neg_item_id,
                    label=0,
                    timestamp=target_timestamp,
                    item_lookup=item_lookup,
                    popularity_lookup=popularity_lookup,
                    cfg=cfg,
                )
                records.append(neg_record)

    return records


def build_eval_samples_for_split(
    train_histories: Dict[str, List[Dict[str, Any]]],
    split_targets: Dict[str, Dict[str, Any]],
    item_lookup: Dict[str, Dict[str, str]],
    popularity_lookup: Dict[str, str],
    cfg: BuildSamplesConfig,
) -> List[Dict[str, Any]]:
    rng = random.Random(cfg.seed)
    all_item_ids = list(item_lookup.keys())
    records: List[Dict[str, Any]] = []

    for user_id, target in split_targets.items():
        if user_id not in train_histories:
            continue

        history_seq = train_histories[user_id]
        history_item_ids = [_normalize_id(x["item_id"]) for x in history_seq if _normalize_id(x["item_id"]) != ""]
        target_item_id = _normalize_id(target["item_id"])
        target_timestamp = target["timestamp"]

        if target_item_id == "" or target_item_id not in item_lookup:
            continue

        if target_item_id in set(history_item_ids):
            continue

        user_seen_items = set(history_item_ids + [target_item_id])

        pos_record = _build_single_record(
            user_id=user_id,
            history_item_ids=history_item_ids,
            candidate_item_id=target_item_id,
            label=1,
            timestamp=target_timestamp,
            item_lookup=item_lookup,
            popularity_lookup=popularity_lookup,
            cfg=cfg,
        )
        records.append(pos_record)

        negative_item_ids = sample_negative_items(
            user_seen_items=user_seen_items,
            all_item_ids=all_item_ids,
            num_negatives=cfg.num_negatives,
            rng=rng,
        )

        for neg_item_id in negative_item_ids:
            if neg_item_id not in item_lookup:
                continue
            if neg_item_id in set(history_item_ids):
                continue

            neg_record = _build_single_record(
                user_id=user_id,
                history_item_ids=history_item_ids,
                candidate_item_id=neg_item_id,
                label=0,
                timestamp=target_timestamp,
                item_lookup=item_lookup,
                popularity_lookup=popularity_lookup,
                cfg=cfg,
            )
            records.append(neg_record)

    return records
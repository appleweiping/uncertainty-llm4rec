# src/data/sample_builder.py

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import pandas as pd

from src.data.candidate_sampling import sample_negative_items


@dataclass
class BuildSamplesConfig:
    max_history_len: int = 10
    num_negatives: int = 5
    seed: int = 42


def normalize_id(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def build_item_lookup(items_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    将 items.csv 转成 item_id -> item信息 的查找表
    期望 items.csv 至少包含:
    item_id, title, candidate_text
    可选:
    categories, description
    """
    required_cols = ["item_id"]
    for col in required_cols:
        if col not in items_df.columns:
            raise ValueError(f"items_df 缺少必要字段: {col}")

    lookup: Dict[str, Dict[str, str]] = {}

    for _, row in items_df.iterrows():
        item_id = normalize_id(row["item_id"])
        lookup[item_id] = {
            "candidate_title": str(row["title"]) if "title" in items_df.columns and pd.notna(row["title"]) else "",
            "candidate_text": str(row["candidate_text"]) if "candidate_text" in items_df.columns and pd.notna(row["candidate_text"]) else "",
            "categories": str(row["categories"]) if "categories" in items_df.columns and pd.notna(row["categories"]) else "",
            "description": str(row["description"]) if "description" in items_df.columns and pd.notna(row["description"]) else "",
        }

    return lookup


def build_popularity_lookup(popularity_df: pd.DataFrame) -> Dict[str, str]:
    """
    将 popularity_stats.csv 转成 item_id -> popularity_group 的查找表
    期望 popularity_df 至少包含:
    item_id, popularity_group
    """
    required_cols = ["item_id", "popularity_group"]
    for col in required_cols:
        if col not in popularity_df.columns:
            raise ValueError(f"popularity_df 缺少必要字段: {col}")

    lookup: Dict[str, str] = {}
    for _, row in popularity_df.iterrows():
        item_id = normalize_id(row["item_id"])
        lookup[item_id] = str(row["popularity_group"]) if pd.notna(row["popularity_group"]) else "mid"
    return lookup


def sort_and_group_interactions(interactions_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    将 interactions.csv 按 user_id + timestamp 排序，并聚成用户序列。
    期望 interactions_df 至少包含:
    user_id, item_id, timestamp
    """
    required_cols = ["user_id", "item_id", "timestamp"]
    for col in required_cols:
        if col not in interactions_df.columns:
            raise ValueError(f"interactions_df 缺少必要字段: {col}")

    df = interactions_df.copy()
    df["user_id"] = df["user_id"].map(normalize_id)
    df["item_id"] = df["item_id"].map(normalize_id)

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    user_sequences: Dict[str, List[Dict[str, Any]]] = {}
    for user_id, group in df.groupby("user_id"):
        seq = []
        for _, row in group.iterrows():
            seq.append(
                {
                    "item_id": normalize_id(row["item_id"]),
                    "timestamp": row["timestamp"],
                    "rating": row["rating"] if "rating" in df.columns else None,
                }
            )
        user_sequences[user_id] = seq

    return user_sequences


def split_user_sequence_leave_one_out(
    user_sequences: Dict[str, List[Dict[str, Any]]]
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    对每个用户做标准 leave-one-out 划分:
    - 最后一个交互作为 test target
    - 倒数第二个交互作为 valid target
    - 更早的交互构成 train history

    Returns:
        train_histories: user -> 历史交互列表（不含 valid/test target）
        valid_targets: user -> valid target record
        test_targets: user -> test target record
    """
    train_histories: Dict[str, List[Dict[str, Any]]] = {}
    valid_targets: Dict[str, Dict[str, Any]] = {}
    test_targets: Dict[str, Dict[str, Any]] = {}

    for user_id, seq in user_sequences.items():
        if len(seq) < 3:
            # 至少需要 train/valid/test 三段
            continue

        train_histories[user_id] = seq[:-2]
        valid_targets[user_id] = seq[-2]
        test_targets[user_id] = seq[-1]

    return train_histories, valid_targets, test_targets


def format_history_text(
    history_item_ids: List[str],
    item_lookup: Dict[str, Dict[str, str]],
    max_history_len: int,
) -> List[str]:
    """
    将用户历史转成文本列表。这里保留列表结构，方便后续 prompt_builder 直接消费。
    每个元素尽量用 title，title 为空则退化到 candidate_text 的前一部分。
    """
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


def build_single_record(
    user_id: str,
    history_item_ids: List[str],
    candidate_item_id: str,
    label: int,
    item_lookup: Dict[str, Dict[str, str]],
    popularity_lookup: Dict[str, str],
    timestamp: Any,
    cfg: BuildSamplesConfig,
) -> Dict[str, Any]:
    item_info = item_lookup.get(candidate_item_id, {})

    return {
        "user_id": user_id,
        "history": format_history_text(history_item_ids, item_lookup, cfg.max_history_len),
        "candidate_item_id": candidate_item_id,
        "candidate_title": item_info.get("candidate_title", ""),
        "candidate_text": item_info.get("candidate_text", ""),
        "label": int(label),
        "target_popularity_group": popularity_lookup.get(candidate_item_id, "mid"),
        "timestamp": timestamp,
    }


def build_eval_samples_for_split(
    train_histories: Dict[str, List[Dict[str, Any]]],
    split_targets: Dict[str, Dict[str, Any]],
    item_lookup: Dict[str, Dict[str, str]],
    popularity_lookup: Dict[str, str],
    cfg: BuildSamplesConfig,
) -> List[Dict[str, Any]]:
    """
    针对 valid/test 构造 pointwise 样本。
    每个用户会产生:
    - 1 条正样本
    - num_negatives 条负样本
    """
    rng = random.Random(cfg.seed)
    all_item_ids = list(item_lookup.keys())

    records: List[Dict[str, Any]] = []

    for user_id, target in split_targets.items():
        if user_id not in train_histories:
            continue

        history_records = train_histories[user_id]
        history_item_ids = [x["item_id"] for x in history_records]
        target_item_id = normalize_id(target["item_id"])
        target_timestamp = target["timestamp"]

        user_seen_items = set(history_item_ids + [target_item_id])

        # 正样本
        pos_record = build_single_record(
            user_id=user_id,
            history_item_ids=history_item_ids,
            candidate_item_id=target_item_id,
            label=1,
            item_lookup=item_lookup,
            popularity_lookup=popularity_lookup,
            timestamp=target_timestamp,
            cfg=cfg,
        )
        records.append(pos_record)

        # 负样本
        negative_item_ids = sample_negative_items(
            user_seen_items=user_seen_items,
            all_item_ids=all_item_ids,
            num_negatives=cfg.num_negatives,
            rng=rng,
        )

        for neg_item_id in negative_item_ids:
            neg_record = build_single_record(
                user_id=user_id,
                history_item_ids=history_item_ids,
                candidate_item_id=neg_item_id,
                label=0,
                item_lookup=item_lookup,
                popularity_lookup=popularity_lookup,
                timestamp=target_timestamp,
                cfg=cfg,
            )
            records.append(neg_record)

    return records


def build_train_pointwise_samples(
    train_histories: Dict[str, List[Dict[str, Any]]],
    item_lookup: Dict[str, Dict[str, str]],
    popularity_lookup: Dict[str, str],
    cfg: BuildSamplesConfig,
) -> List[Dict[str, Any]]:
    """
    构造训练集 pointwise 样本。
    对于 train history 中从第二个交互开始，每个位置都可以构造:
    - 当前 item 的正样本
    - 若干负样本
    """
    rng = random.Random(cfg.seed)
    all_item_ids = list(item_lookup.keys())

    records: List[Dict[str, Any]] = []

    for user_id, seq in train_histories.items():
        if len(seq) < 2:
            continue

        full_train_item_ids = [x["item_id"] for x in seq]

        for idx in range(1, len(seq)):
            history_records = seq[:idx]
            history_item_ids = [x["item_id"] for x in history_records]

            target = seq[idx]
            target_item_id = normalize_id(target["item_id"])
            target_timestamp = target["timestamp"]

            user_seen_items = set(full_train_item_ids)

            pos_record = build_single_record(
                user_id=user_id,
                history_item_ids=history_item_ids,
                candidate_item_id=target_item_id,
                label=1,
                item_lookup=item_lookup,
                popularity_lookup=popularity_lookup,
                timestamp=target_timestamp,
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
                neg_record = build_single_record(
                    user_id=user_id,
                    history_item_ids=history_item_ids,
                    candidate_item_id=neg_item_id,
                    label=0,
                    item_lookup=item_lookup,
                    popularity_lookup=popularity_lookup,
                    timestamp=target_timestamp,
                    cfg=cfg,
                )
                records.append(neg_record)

    return records
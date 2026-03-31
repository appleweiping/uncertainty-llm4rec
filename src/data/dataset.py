from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.utils.io import load_jsonl


def load_samples(path: str | Path) -> List[Dict[str, Any]]:
    """
    读取 jsonl 样本文件（train/valid/test/predictions 都可以）
    """
    return load_jsonl(path)


def build_target_map(samples: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    构建 user -> target_item_id 映射

    注意:pointwise 数据中同一个 user 会重复出现，
    但 target_item_id 是一致的，所以可以直接覆盖写。
    """
    target_map: Dict[str, str] = {}

    for row in samples:
        user_id = str(row["user_id"])

        if "target_item_id" in row:
            target_map[user_id] = str(row["target_item_id"])
        elif "target_item" in row:
            target_map[user_id] = str(row["target_item"]["item_id"])
        else:
            raise KeyError("Sample missing target_item_id")

    return target_map


def group_samples_by_user(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    将 pointwise 样本按 user 聚合

    输出:
    {
        user_id: [
            {candidate1},
            {candidate2},
            ...
        ]
    }

    👉 这是后面 reranking / evaluation 必须用的结构
    """
    user_groups: Dict[str, List[Dict[str, Any]]] = {}

    for row in samples:
        user_id = str(row["user_id"])
        user_groups.setdefault(user_id, []).append(row)

    return user_groups


def extract_user_candidates(
    samples: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    提取每个 user 的 candidate 列表（带 label / text 等）

    输出结构:
    {
        user_id: [
            {
                "candidate_item_id": ...,
                "label": ...,
                "candidate_text": ...,
                ...
            }
        ]
    }

    👉 这个函数是给 inference / rerank / eval 用的核心接口
    """
    user_groups = group_samples_by_user(samples)

    result: Dict[str, List[Dict[str, Any]]] = {}

    for user_id, rows in user_groups.items():
        candidates = []

        for r in rows:
            candidates.append(
                {
                    "candidate_item_id": str(r["candidate_item_id"]),
                    "label": int(r["label"]),
                    "candidate_text": r.get("candidate_text", ""),
                    "candidate_title": r.get("candidate_title", ""),
                    "target_popularity_group": r.get("target_popularity_group", "unknown"),
                }
            )

        result[user_id] = candidates

    return result


def extract_user_history(
    samples: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    提取每个 user 的历史（只取第一条即可，因为重复）

    输出:
    {
        user_id: history_list
    }
    """
    user_history: Dict[str, List[Dict[str, Any]]] = {}

    for row in samples:
        user_id = str(row["user_id"])
        if user_id not in user_history:
            user_history[user_id] = row.get("history", [])

    return user_history
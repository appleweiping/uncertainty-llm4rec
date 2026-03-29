# src/data/dataset.py
# src/data/dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl


def load_samples(path: str | Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def build_target_map(samples_or_predictions: list[dict[str, Any]]) -> dict[str, str]:
    target_map: dict[str, str] = {}
    for row in samples_or_predictions:
        user_id = row["user_id"]
        if "target_item" in row:
            target_map[user_id] = row["target_item"]["item_id"]
        elif "target_item_id" in row:
            target_map[user_id] = row["target_item_id"]
    return target_map
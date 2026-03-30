# src/data/noise.py
from __future__ import annotations

import copy
import random
from typing import Any


def perturb_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    replacements = [
        ("science fiction", "sci-fi"),
        ("thriller", "suspense"),
        ("film", "movie"),
        ("story", "plot"),
        ("romantic", "love-themed"),
        ("contemporary", "modern"),
    ]

    out = text
    for src, tgt in replacements:
        out = out.replace(src, tgt)
    return out


def perturb_history_items(
    history_items: list[dict[str, Any]],
    drop_prob: float = 0.2,
) -> list[dict[str, Any]]:
    if not isinstance(history_items, list):
        return history_items

    kept = [item for item in history_items if random.random() > drop_prob]

    if len(kept) == 0 and len(history_items) > 0:
        kept = [copy.deepcopy(random.choice(history_items))]

    return kept


def perturb_item(item: dict[str, Any], text_noise_prob: float = 0.5) -> dict[str, Any]:
    if not isinstance(item, dict):
        return item

    out = copy.deepcopy(item)
    if "meta" in out and isinstance(out["meta"], str) and random.random() < text_noise_prob:
        out["meta"] = perturb_text(out["meta"])
    if "title" in out and isinstance(out["title"], str) and random.random() < text_noise_prob * 0.3:
        out["title"] = out["title"]
    return out


def perturb_candidates(
    candidates: list[dict[str, Any]],
    text_noise_prob: float = 0.5,
) -> list[dict[str, Any]]:
    if not isinstance(candidates, list):
        return candidates
    return [perturb_item(item, text_noise_prob=text_noise_prob) for item in candidates]


def apply_noise_to_sample(
    sample: dict[str, Any],
    history_drop_prob: float = 0.2,
    text_noise_prob: float = 0.5,
    label_flip_prob: float = 0.0,
) -> dict[str, Any]:
    s = copy.deepcopy(sample)

    if "history_items" in s:
        s["history_items"] = perturb_history_items(
            s["history_items"],
            drop_prob=history_drop_prob,
        )

    if "target_item" in s:
        s["target_item"] = perturb_item(
            s["target_item"],
            text_noise_prob=text_noise_prob,
        )

    if "candidates" in s:
        s["candidates"] = perturb_candidates(
            s["candidates"],
            text_noise_prob=text_noise_prob,
        )

    if "label" in s and random.random() < label_flip_prob:
        try:
            s["label"] = 1 - int(s["label"])
        except Exception:
            pass

    return s
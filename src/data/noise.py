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
        ("anti aging", "age-defying"),
        ("hydrating", "moisturizing"),
        ("fragrance free", "unscented"),
    ]

    out = text
    for src, tgt in replacements:
        out = out.replace(src, tgt)
        out = out.replace(src.title(), tgt.title())
    return out


def inject_history_noise(
    history: list[Any],
    drop_prob: float = 0.2,
) -> list[Any]:
    if not isinstance(history, list):
        return history

    kept = [copy.deepcopy(item) for item in history if random.random() > drop_prob]
    if len(kept) == 0 and len(history) > 0:
        kept = [copy.deepcopy(random.choice(history))]
    return kept


def perturb_history_items(
    history_items: list[dict[str, Any]],
    drop_prob: float = 0.2,
) -> list[dict[str, Any]]:
    return inject_history_noise(history_items, drop_prob=drop_prob)


def perturb_item(item: dict[str, Any], text_noise_prob: float = 0.5) -> dict[str, Any]:
    if not isinstance(item, dict):
        return item

    out = copy.deepcopy(item)
    for key in ["meta", "title", "candidate_text", "candidate_title"]:
        if key in out and isinstance(out[key], str) and random.random() < text_noise_prob:
            out[key] = perturb_text(out[key])
    return out


def perturb_candidates(
    candidates: list[dict[str, Any]],
    text_noise_prob: float = 0.5,
) -> list[dict[str, Any]]:
    if not isinstance(candidates, list):
        return candidates
    return [perturb_item(item, text_noise_prob=text_noise_prob) for item in candidates]


def perturb_item_text(
    sample: dict[str, Any],
    text_noise_prob: float = 0.5,
) -> dict[str, Any]:
    if not isinstance(sample, dict):
        return sample

    out = copy.deepcopy(sample)

    if "target_item" in out and isinstance(out["target_item"], dict):
        out["target_item"] = perturb_item(out["target_item"], text_noise_prob=text_noise_prob)
    if "candidates" in out:
        out["candidates"] = perturb_candidates(out["candidates"], text_noise_prob=text_noise_prob)

    for key in ["candidate_text", "candidate_title", "target_text", "target_title"]:
        if key in out and isinstance(out[key], str) and random.random() < text_noise_prob:
            out[key] = perturb_text(out[key])

    return out


def flip_labels_with_prob(label: Any, flip_prob: float = 0.0) -> Any:
    if random.random() >= flip_prob:
        return label

    try:
        return 1 - int(label)
    except Exception:
        return label


def apply_noise_to_sample(
    sample: dict[str, Any],
    history_drop_prob: float = 0.2,
    text_noise_prob: float = 0.5,
    label_flip_prob: float = 0.0,
) -> dict[str, Any]:
    s = copy.deepcopy(sample)

    if "history" in s:
        s["history"] = inject_history_noise(
            s["history"],
            drop_prob=history_drop_prob,
        )

    if "history_items" in s:
        s["history_items"] = perturb_history_items(
            s["history_items"],
            drop_prob=history_drop_prob,
        )

    s = perturb_item_text(s, text_noise_prob=text_noise_prob)

    if "label" in s:
        s["label"] = flip_labels_with_prob(s["label"], flip_prob=label_flip_prob)

    return s

# src/data/noise.py

from __future__ import annotations
import random
import copy


def perturb_user_history(history, drop_prob=0.3):
    """
    随机删除用户历史中的一部分
    """
    if not isinstance(history, list):
        return history

    return [h for h in history if random.random() > drop_prob]


def perturb_item_text(text: str):
    """
    模拟文本噪声：简化/模糊表达
    """
    if not isinstance(text, str):
        return text

    replacements = [
        ("science fiction", "sci-fi"),
        ("thriller", "suspense"),
        ("film", "movie"),
        ("story", "plot"),
    ]

    for k, v in replacements:
        text = text.replace(k, v)

    return text


def perturb_label(label, flip_prob=0.1):
    """
    标签扰动（可选）
    """
    if random.random() < flip_prob:
        return 1 - label
    return label


def apply_noise_to_sample(sample: dict):
    s = copy.deepcopy(sample)

    # history noise
    if "history" in s:
        s["history"] = perturb_user_history(s["history"])

    # text noise
    if "prompt" in s:
        s["prompt"] = perturb_item_text(s["prompt"])

    # label noise
    if "label" in s:
        s["label"] = perturb_label(s["label"])

    return s
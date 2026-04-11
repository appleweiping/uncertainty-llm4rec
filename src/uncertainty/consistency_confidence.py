# src/uncertainty/consistency_confidence.py

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _extract_votes(predictions: Iterable[dict]) -> list[int]:
    votes: list[int] = []
    for pred in predictions:
        recommend = str(pred.get("recommend", "unknown")).strip().lower()
        if recommend == "yes":
            votes.append(1)
        elif recommend == "no":
            votes.append(0)
    return votes


def _extract_confidences(predictions: Iterable[dict]) -> list[float]:
    confidences: list[float] = []
    for pred in predictions:
        value = pred.get("confidence", -1.0)
        try:
            conf = float(value)
        except Exception:
            continue
        if conf >= 0.0:
            confidences.append(max(0.0, min(1.0, conf)))
    return confidences


def compute_yes_ratio(predictions: Iterable[dict]) -> float:
    votes = _extract_votes(predictions)
    if not votes:
        return 0.5
    return float(np.mean(votes))


def compute_vote_entropy(predictions: Iterable[dict]) -> float:
    yes_ratio = compute_yes_ratio(predictions)
    no_ratio = 1.0 - yes_ratio

    entropy = 0.0
    for p in (yes_ratio, no_ratio):
        if p > 0.0:
            entropy -= p * math.log2(p)

    # Binary entropy is in [0, 1] when using log base 2.
    return float(entropy)


def compute_vote_variance(predictions: Iterable[dict]) -> float:
    votes = _extract_votes(predictions)
    if not votes:
        return 0.25
    return float(np.var(votes))


def compute_mean_confidence(predictions: Iterable[dict]) -> float:
    confidences = _extract_confidences(predictions)
    if not confidences:
        return 0.5
    return float(np.mean(confidences))


def compute_confidence_variance(predictions: Iterable[dict]) -> float:
    confidences = _extract_confidences(predictions)
    if not confidences:
        return 0.0
    return float(np.var(confidences))


def compute_consistency_confidence(predictions: Iterable[dict]) -> float:
    yes_ratio = compute_yes_ratio(predictions)
    no_ratio = 1.0 - yes_ratio
    return float(max(yes_ratio, no_ratio))


def compute_consistency_uncertainty(predictions: Iterable[dict]) -> float:
    return compute_vote_entropy(predictions)


def compute_consistency_summary(predictions: Iterable[dict]) -> dict[str, float | int | str]:
    predictions = list(predictions)
    total_runs = int(len(predictions))
    votes = _extract_votes(predictions)

    yes_count = int(sum(votes))
    no_count = int(len(votes) - yes_count)
    unknown_count = int(total_runs - len(votes))

    yes_ratio = compute_yes_ratio(predictions)
    no_ratio = float(1.0 - yes_ratio)
    majority_vote = "yes" if yes_ratio >= 0.5 else "no"
    majority_ratio = float(max(yes_ratio, no_ratio))

    return {
        "num_consistency_samples": total_runs,
        "yes_count": yes_count,
        "no_count": no_count,
        "unknown_count": unknown_count,
        "yes_ratio": yes_ratio,
        "no_ratio": no_ratio,
        "unknown_ratio": float(unknown_count / total_runs) if total_runs > 0 else 0.0,
        "majority_vote": majority_vote,
        "majority_ratio": majority_ratio,
        "vote_entropy": compute_vote_entropy(predictions),
        "vote_variance": compute_vote_variance(predictions),
        "mean_confidence": compute_mean_confidence(predictions),
        "confidence_variance": compute_confidence_variance(predictions),
        "consistency_confidence": compute_consistency_confidence(predictions),
        "consistency_uncertainty": compute_consistency_uncertainty(predictions),
    }

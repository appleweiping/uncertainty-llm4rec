"""Calibration metrics for confidence-bearing recommendation predictions."""

from __future__ import annotations

from typing import Any

from llm4rec.metrics.confidence import CORRECTNESS_TARGET, _confidence_rows, validate_confidence_value


def calibration_metrics(predictions: list[dict[str, Any]], *, bins: int = 10) -> dict[str, Any]:
    rows, missing = _confidence_rows(predictions)
    return {
        "correctness_target": CORRECTNESS_TARGET,
        "ece": expected_calibration_error(rows, bins=bins),
        "brier_score": brier_score(rows),
        "confidence_bucket_stats": confidence_bucket_stats(rows, bins=bins),
        "reliability_diagram_data": reliability_diagram_data(rows, bins=bins),
        "count_missing_confidence": missing,
    }


def expected_calibration_error(rows: list[dict[str, Any]], *, bins: int = 10) -> float:
    if not rows:
        return 0.0
    stats = confidence_bucket_stats(rows, bins=bins)
    total = len(rows)
    return sum(
        bucket["count"] / total * abs(bucket["accuracy"] - bucket["mean_confidence"])
        for bucket in stats
        if bucket["count"]
    )


def brier_score(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum((validate_confidence_value(row["confidence"]) - float(row["correct"])) ** 2 for row in rows) / len(rows)


def confidence_bucket_stats(rows: list[dict[str, Any]], *, bins: int = 10) -> list[dict[str, Any]]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    buckets = [
        {
            "bin": index,
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "count": 0,
            "accuracy": 0.0,
            "mean_confidence": 0.0,
        }
        for index in range(bins)
    ]
    for row in rows:
        confidence = validate_confidence_value(row["confidence"])
        index = min(bins - 1, int(confidence * bins))
        bucket = buckets[index]
        bucket["count"] += 1
        bucket["accuracy"] += float(row["correct"])
        bucket["mean_confidence"] += confidence
    for bucket in buckets:
        if bucket["count"]:
            bucket["accuracy"] /= bucket["count"]
            bucket["mean_confidence"] /= bucket["count"]
    return buckets


def reliability_diagram_data(rows: list[dict[str, Any]], *, bins: int = 10) -> list[dict[str, Any]]:
    return confidence_bucket_stats(rows, bins=bins)

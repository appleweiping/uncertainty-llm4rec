"""Long-tail metrics using train-only item popularity buckets."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any

from llm4rec.metrics.confidence import validate_confidence_value
from llm4rec.metrics.ranking import dedupe

DEFAULT_BUCKET_FRACTIONS = {"head": 0.20, "mid": 0.30, "tail": 0.50}
BUCKET_ORDER = ["head", "mid", "tail"]


def assign_popularity_buckets(
    train_popularity: dict[str, int],
    *,
    catalog_items: list[str],
    bucket_fractions: dict[str, float] | None = None,
) -> dict[str, str]:
    fractions = bucket_fractions or DEFAULT_BUCKET_FRACTIONS
    items = sorted(set(catalog_items) | set(train_popularity), key=lambda item: (-train_popularity.get(item, 0), item))
    if not items:
        return {}
    head_count = max(1, math.ceil(len(items) * float(fractions.get("head", 0.20))))
    mid_count = max(1, math.ceil(len(items) * float(fractions.get("mid", 0.30)))) if len(items) > 1 else 0
    buckets: dict[str, str] = {}
    for index, item in enumerate(items):
        if index < head_count:
            bucket = "head"
        elif index < head_count + mid_count:
            bucket = "mid"
        else:
            bucket = "tail"
        buckets[item] = bucket
    if "tail" not in set(buckets.values()) and len(items) >= 3:
        buckets[items[-1]] = "tail"
    return buckets


def long_tail_metrics(
    predictions: list[dict[str, Any]],
    *,
    train_popularity: dict[str, int],
    catalog_items: list[str],
    top_k: list[int],
    bucket_fractions: dict[str, float] | None = None,
) -> dict[str, Any]:
    item_buckets = assign_popularity_buckets(
        train_popularity,
        catalog_items=catalog_items,
        bucket_fractions=bucket_fractions,
    )
    metrics: dict[str, Any] = {
        "popularity_source": "train_examples_only",
        "bucket_fractions": bucket_fractions or DEFAULT_BUCKET_FRACTIONS,
        "item_bucket_counts": _bucket_counts(item_buckets),
    }
    for k in sorted({int(value) for value in top_k}):
        metrics[f"hit_rate_by_popularity_bucket@{k}"] = hit_rate_by_popularity_bucket(
            predictions,
            item_buckets=item_buckets,
            k=k,
        )
        metrics[f"recall_by_popularity_bucket@{k}"] = recall_by_popularity_bucket(
            predictions,
            item_buckets=item_buckets,
            k=k,
        )
        metrics[f"coverage_by_popularity_bucket@{k}"] = coverage_by_popularity_bucket(
            predictions,
            item_buckets=item_buckets,
            k=k,
        )
    confidence = confidence_by_popularity_bucket(predictions, item_buckets=item_buckets)
    if confidence:
        metrics["confidence_by_popularity_bucket"] = confidence
    return metrics


def hit_rate_by_popularity_bucket(
    predictions: list[dict[str, Any]],
    *,
    item_buckets: dict[str, str],
    k: int,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in predictions:
        bucket = item_buckets.get(str(row.get("target_item")), "unknown")
        hit = str(row.get("target_item")) in dedupe(str(item) for item in row.get("predicted_items", []))[:k]
        grouped[bucket].append(float(hit))
    return {bucket: mean(values) for bucket, values in sorted(grouped.items())}


def recall_by_popularity_bucket(
    predictions: list[dict[str, Any]],
    *,
    item_buckets: dict[str, str],
    k: int,
) -> dict[str, float]:
    return hit_rate_by_popularity_bucket(predictions, item_buckets=item_buckets, k=k)


def coverage_by_popularity_bucket(
    predictions: list[dict[str, Any]],
    *,
    item_buckets: dict[str, str],
    k: int,
) -> dict[str, float]:
    bucket_items: dict[str, set[str]] = defaultdict(set)
    recommended: dict[str, set[str]] = defaultdict(set)
    for item, bucket in item_buckets.items():
        bucket_items[bucket].add(item)
    for row in predictions:
        for item in dedupe(str(value) for value in row.get("predicted_items", []))[:k]:
            bucket = item_buckets.get(item, "unknown")
            recommended[bucket].add(item)
    return {
        bucket: len(recommended.get(bucket, set())) / len(items) if items else 0.0
        for bucket, items in sorted(bucket_items.items())
    }


def confidence_by_popularity_bucket(
    predictions: list[dict[str, Any]],
    *,
    item_buckets: dict[str, str],
) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in predictions:
        metadata = row.get("metadata") or {}
        if metadata.get("confidence") is None:
            continue
        bucket = item_buckets.get(str(row.get("target_item")), "unknown")
        grouped[bucket].append(validate_confidence_value(metadata["confidence"]))
    return {
        bucket: {"count": len(values), "mean_confidence": mean(values) if values else None}
        for bucket, values in sorted(grouped.items())
    }


def _bucket_counts(item_buckets: dict[str, str]) -> dict[str, int]:
    return {bucket: sum(1 for value in item_buckets.values() if value == bucket) for bucket in BUCKET_ORDER}

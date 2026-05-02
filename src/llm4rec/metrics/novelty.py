"""Novelty metrics based only on train-set item popularity."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from llm4rec.metrics.ranking import dedupe


def train_item_popularity(train_examples: list[dict[str, Any]] | None) -> dict[str, int]:
    """Count train-observed items from train histories and train targets."""
    counts: Counter[str] = Counter()
    for row in train_examples or []:
        for item in row.get("history", []):
            counts[str(item)] += 1
        target = row.get("target") or row.get("target_item")
        if target is not None:
            counts[str(target)] += 1
    return dict(counts)


def mean_self_information(
    predictions: list[dict[str, Any]],
    *,
    train_popularity: dict[str, int],
    catalog_items: list[str],
    k: int,
) -> float:
    total = sum(train_popularity.values())
    denominator = total + max(len(set(catalog_items)), 1)
    values = [
        _self_information_with_denominator(item, train_popularity=train_popularity, denominator=denominator)
        for row in predictions
        for item in dedupe(str(value) for value in row.get("predicted_items", []))[:k]
    ]
    return sum(values) / len(values) if values else 0.0


def average_popularity_rank(
    predictions: list[dict[str, Any]],
    *,
    train_popularity: dict[str, int],
    catalog_items: list[str],
    k: int,
) -> float:
    ranks = popularity_ranks(train_popularity, catalog_items=catalog_items)
    fallback = len(ranks) + 1
    values = [
        float(ranks.get(item, fallback))
        for row in predictions
        for item in dedupe(str(value) for value in row.get("predicted_items", []))[:k]
    ]
    return sum(values) / len(values) if values else 0.0


def novelty_at_k(
    predictions: list[dict[str, Any]],
    *,
    train_popularity: dict[str, int],
    catalog_items: list[str],
    k: int,
) -> float:
    return mean_self_information(predictions, train_popularity=train_popularity, catalog_items=catalog_items, k=k)


def novelty_metrics(
    predictions: list[dict[str, Any]],
    *,
    train_popularity: dict[str, int],
    catalog_items: list[str],
    top_k: list[int],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "popularity_source": "train_examples_only",
        "train_popularity_item_count": len(train_popularity),
    }
    for k in sorted({int(value) for value in top_k}):
        metrics[f"mean_self_information@{k}"] = mean_self_information(
            predictions,
            train_popularity=train_popularity,
            catalog_items=catalog_items,
            k=k,
        )
        metrics[f"average_popularity_rank@{k}"] = average_popularity_rank(
            predictions,
            train_popularity=train_popularity,
            catalog_items=catalog_items,
            k=k,
        )
        metrics[f"novelty@{k}"] = novelty_at_k(
            predictions,
            train_popularity=train_popularity,
            catalog_items=catalog_items,
            k=k,
        )
    return metrics


def popularity_ranks(train_popularity: dict[str, int], *, catalog_items: list[str]) -> dict[str, int]:
    items = sorted(set(catalog_items) | set(train_popularity), key=lambda item: (-train_popularity.get(item, 0), item))
    return {item: rank for rank, item in enumerate(items, start=1)}


def _self_information(item: str, *, train_popularity: dict[str, int], catalog_size: int) -> float:
    total = sum(train_popularity.values())
    denominator = total + max(catalog_size, 1)
    return _self_information_with_denominator(
        item,
        train_popularity=train_popularity,
        denominator=denominator,
    )


def _self_information_with_denominator(
    item: str,
    *,
    train_popularity: dict[str, int],
    denominator: int,
) -> float:
    probability = (train_popularity.get(item, 0) + 1) / denominator
    return -math.log2(probability)

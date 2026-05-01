"""Diversity metrics using item metadata fallback signals."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from llm4rec.metrics.ranking import dedupe


def item_category_map(item_catalog: list[dict[str, Any]] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in item_catalog or []:
        item_id = str(row.get("item_id") or "")
        if not item_id:
            continue
        category = str(row.get("category") or row.get("domain") or "unknown")
        mapping[item_id] = category
    return mapping


def intra_list_diversity(
    predictions: list[dict[str, Any]],
    *,
    item_categories: dict[str, str],
    k: int,
) -> float:
    """Average pairwise category dissimilarity within each top-k list."""
    values = [_row_intra_list_diversity(row, item_categories=item_categories, k=k) for row in predictions]
    return sum(values) / len(values) if values else 0.0


def unique_category_count_at_k(
    predictions: list[dict[str, Any]],
    *,
    item_categories: dict[str, str],
    k: int,
) -> float:
    values = []
    for row in predictions:
        categories = _top_categories(row, item_categories=item_categories, k=k)
        values.append(float(len(set(categories))))
    return sum(values) / len(values) if values else 0.0


def category_entropy_at_k(
    predictions: list[dict[str, Any]],
    *,
    item_categories: dict[str, str],
    k: int,
) -> float:
    values = []
    for row in predictions:
        categories = _top_categories(row, item_categories=item_categories, k=k)
        if not categories:
            values.append(0.0)
            continue
        counts = Counter(categories)
        total = len(categories)
        values.append(-sum((count / total) * math.log2(count / total) for count in counts.values()))
    return sum(values) / len(values) if values else 0.0


def diversity_metrics(
    predictions: list[dict[str, Any]],
    *,
    item_catalog: list[dict[str, Any]] | None,
    top_k: list[int],
) -> dict[str, Any]:
    categories = item_category_map(item_catalog)
    metrics: dict[str, Any] = {
        "diversity_signal": "item_category",
        "embedding_diversity_available": False,
    }
    for k in sorted({int(value) for value in top_k}):
        metrics[f"intra_list_diversity@{k}"] = intra_list_diversity(predictions, item_categories=categories, k=k)
        metrics[f"unique_category_count@{k}"] = unique_category_count_at_k(predictions, item_categories=categories, k=k)
        metrics[f"category_entropy@{k}"] = category_entropy_at_k(predictions, item_categories=categories, k=k)
    return metrics


def _row_intra_list_diversity(row: dict[str, Any], *, item_categories: dict[str, str], k: int) -> float:
    categories = _top_categories(row, item_categories=item_categories, k=k)
    if len(categories) < 2:
        return 0.0
    pair_count = 0
    dissimilar = 0
    for left_index, left in enumerate(categories):
        for right in categories[left_index + 1:]:
            pair_count += 1
            dissimilar += int(left != right)
    return dissimilar / pair_count if pair_count else 0.0


def _top_categories(row: dict[str, Any], *, item_categories: dict[str, str], k: int) -> list[str]:
    items = dedupe(str(item) for item in row.get("predicted_items", []))[:k]
    return [item_categories.get(item, "unknown") for item in items]

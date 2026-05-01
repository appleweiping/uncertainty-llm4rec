"""Ranking metrics for JSONL prediction records."""

from __future__ import annotations

import math
from typing import Any, Iterable


def dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def ranking_metrics(predictions: list[dict[str, Any]], *, top_k: list[int]) -> dict[str, Any]:
    if not top_k:
        raise ValueError("top_k must not be empty")
    metrics: dict[str, Any] = {}
    for k in sorted({int(value) for value in top_k}):
        if k < 1:
            raise ValueError("top_k values must be >= 1")
        metrics[f"recall@{k}"] = _mean(_recall_at_k(row, k) for row in predictions)
        metrics[f"hit_rate@{k}"] = _mean(_hit_at_k(row, k) for row in predictions)
        metrics[f"mrr@{k}"] = _mean(_mrr_at_k(row, k) for row in predictions)
        metrics[f"ndcg@{k}"] = _mean(_ndcg_at_k(row, k) for row in predictions)
        metrics[f"coverage@{k}"] = coverage(predictions, k=k)["coverage_rate"]
    metrics["coverage"] = coverage(predictions)["coverage_rate"]
    metrics["evaluated_count"] = len(predictions)
    return metrics


def coverage(predictions: list[dict[str, Any]], *, k: int | None = None) -> dict[str, Any]:
    catalog = {
        item
        for row in predictions
        for item in row.get("candidate_items", [])
    }
    recommended: set[str] = set()
    for row in predictions:
        items = dedupe(row.get("predicted_items", []))
        if k is not None:
            items = items[:k]
        recommended.update(items)
    denominator = len(catalog)
    return {
        "unique_recommended": len(recommended),
        "catalog_size": denominator,
        "coverage_rate": (len(recommended) / denominator) if denominator else 0.0,
    }


def _recall_at_k(row: dict[str, Any], k: int) -> float:
    return 1.0 if row["target_item"] in dedupe(row.get("predicted_items", []))[:k] else 0.0


def _hit_at_k(row: dict[str, Any], k: int) -> float:
    return _recall_at_k(row, k)


def _mrr_at_k(row: dict[str, Any], k: int) -> float:
    target = row["target_item"]
    for rank, item in enumerate(dedupe(row.get("predicted_items", []))[:k], start=1):
        if item == target:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(row: dict[str, Any], k: int) -> float:
    target = row["target_item"]
    for rank, item in enumerate(dedupe(row.get("predicted_items", []))[:k], start=1):
        if item == target:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return sum(rows) / len(rows)

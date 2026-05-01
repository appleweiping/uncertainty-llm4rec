"""Evaluation slice construction for Phase 5 metrics."""

from __future__ import annotations

from typing import Any

from llm4rec.metrics.cold_start import metadata_availability_bucket, user_history_bucket
from llm4rec.metrics.long_tail import assign_popularity_buckets


def enrich_predictions_for_slices(
    predictions: list[dict[str, Any]],
    *,
    examples: list[dict[str, Any]] | None = None,
    item_catalog: list[dict[str, Any]] | None = None,
    train_popularity: dict[str, int] | None = None,
    catalog_items: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Attach evaluation-only slice metadata without changing labels or splits."""
    example_by_id = {
        str(row.get("example_id")): row
        for row in examples or []
        if row.get("example_id") is not None
    }
    item_by_id = {
        str(row.get("item_id")): row
        for row in item_catalog or []
        if row.get("item_id") is not None
    }
    item_buckets = assign_popularity_buckets(
        train_popularity or {},
        catalog_items=catalog_items or sorted(item_by_id),
    )
    enriched: list[dict[str, Any]] = []
    for row in predictions:
        copy = dict(row)
        metadata = dict(copy.get("metadata") or {})
        example = example_by_id.get(str(metadata.get("example_id") or ""))
        history = example.get("history", []) if isinstance(example, dict) else None
        if history is not None:
            metadata.setdefault("history_length", len(history))
        history_length = int(metadata.get("history_length") or 0)
        metadata.setdefault("user_history_bucket", user_history_bucket(history_length))
        target = str(copy.get("target_item") or "")
        metadata.setdefault("target_train_popularity", int((train_popularity or {}).get(target, 0)))
        metadata.setdefault("target_item_popularity_bucket", item_buckets.get(target, "unknown"))
        metadata.setdefault("item_metadata_bucket", metadata_availability_bucket(item_by_id.get(target)))
        copy["metadata"] = metadata
        enriched.append(copy)
    return enriched

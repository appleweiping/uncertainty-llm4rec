"""Generic deterministic metric slicing utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


def slice_predictions(
    predictions: list[dict[str, Any]],
    *,
    key: str,
    default: str = "unknown",
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        value = _lookup(row, key)
        grouped[str(value if value not in (None, "") else default)].append(row)
    return dict(sorted(grouped.items()))


def slice_predictions_by(
    predictions: list[dict[str, Any]],
    *,
    name: str,
    value_fn: Callable[[dict[str, Any]], str],
) -> dict[str, list[dict[str, Any]]]:
    del name
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        value = value_fn(row)
        grouped[str(value or "unknown")].append(row)
    return dict(sorted(grouped.items()))


def _lookup(row: dict[str, Any], dotted_key: str) -> Any:
    value: Any = row
    for part in dotted_key.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value

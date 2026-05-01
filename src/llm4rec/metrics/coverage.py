"""Coverage metrics for candidate-grounded recommendation outputs."""

from __future__ import annotations

from typing import Any, Iterable

from llm4rec.metrics.ranking import dedupe


def item_coverage_at_k(predictions: list[dict[str, Any]], *, k: int) -> dict[str, Any]:
    """Return unique recommended item count and rate over the visible catalog."""
    _validate_k(k)
    catalog = _visible_catalog(predictions)
    recommended = _recommended_items(predictions, k=k)
    return {
        "k": k,
        "unique_recommended": len(recommended),
        "catalog_size": len(catalog),
        "coverage_rate": len(recommended) / len(catalog) if catalog else 0.0,
    }


def catalog_coverage_at_k(
    predictions: list[dict[str, Any]],
    *,
    catalog_items: Iterable[str] | None = None,
    k: int,
) -> dict[str, Any]:
    """Return coverage over a supplied catalog, or over visible candidates."""
    _validate_k(k)
    catalog = {str(item) for item in catalog_items} if catalog_items is not None else _visible_catalog(predictions)
    recommended = _recommended_items(predictions, k=k)
    return {
        "k": k,
        "unique_recommended": len(recommended),
        "catalog_size": len(catalog),
        "coverage_rate": len(recommended & catalog) / len(catalog) if catalog else 0.0,
    }


def user_coverage(predictions: list[dict[str, Any]], *, k: int | None = None) -> dict[str, Any]:
    """Fraction of users with at least one recommendation."""
    if k is not None:
        _validate_k(k)
    covered = 0
    users: set[str] = set()
    for row in predictions:
        users.add(str(row.get("user_id") or ""))
        items = dedupe(str(item) for item in row.get("predicted_items", []))
        if k is not None:
            items = items[:k]
        if items:
            covered += 1
    total = len(predictions)
    return {
        "covered_user_rows": covered,
        "user_row_count": total,
        "unique_users": len(users - {""}),
        "user_coverage_rate": covered / total if total else 0.0,
    }


def coverage_by_domain(predictions: list[dict[str, Any]], *, k: int) -> dict[str, Any]:
    """Compute item coverage for each domain present in predictions."""
    _validate_k(k)
    output: dict[str, Any] = {}
    for domain in sorted({str(row.get("domain") or "unknown") for row in predictions}):
        rows = [row for row in predictions if str(row.get("domain") or "unknown") == domain]
        output[domain] = item_coverage_at_k(rows, k=k)
    return output


def coverage_metrics(
    predictions: list[dict[str, Any]],
    *,
    top_k: list[int],
    catalog_items: Iterable[str] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"user_coverage": user_coverage(predictions)}
    for k in sorted({int(value) for value in top_k}):
        metrics[f"item_coverage@{k}"] = item_coverage_at_k(predictions, k=k)
        metrics[f"catalog_coverage@{k}"] = catalog_coverage_at_k(predictions, catalog_items=catalog_items, k=k)
        metrics[f"user_coverage@{k}"] = user_coverage(predictions, k=k)
    return metrics


def _visible_catalog(predictions: list[dict[str, Any]]) -> set[str]:
    return {
        str(item)
        for row in predictions
        for item in row.get("candidate_items", [])
    }


def _recommended_items(predictions: list[dict[str, Any]], *, k: int) -> set[str]:
    recommended: set[str] = set()
    for row in predictions:
        recommended.update(dedupe(str(item) for item in row.get("predicted_items", []))[:k])
    return recommended


def _validate_k(k: int) -> None:
    if int(k) < 1:
        raise ValueError("k must be >= 1")

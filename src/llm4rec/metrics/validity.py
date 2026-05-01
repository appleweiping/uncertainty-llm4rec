"""Validity and hallucination metrics for candidate-grounded predictions."""

from __future__ import annotations

from typing import Any

from llm4rec.metrics.ranking import dedupe


def validity_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {
            "validity_rate": 0.0,
            "hallucination_rate": 0.0,
            "valid_prediction_count": 0,
            "hallucinated_prediction_count": 0,
        }
    valid_rows = 0
    hallucinated_rows = 0
    for row in predictions:
        candidates = set(row.get("candidate_items", []))
        predicted = dedupe(row.get("predicted_items", []))
        invalid = [item for item in predicted if item not in candidates]
        if predicted and not invalid:
            valid_rows += 1
        if invalid:
            hallucinated_rows += 1
    total = len(predictions)
    return {
        "validity_rate": valid_rows / total,
        "hallucination_rate": hallucinated_rows / total,
        "valid_prediction_count": valid_rows,
        "hallucinated_prediction_count": hallucinated_rows,
    }

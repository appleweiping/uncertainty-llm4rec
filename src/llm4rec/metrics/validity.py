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
            "grounding_success_rate": 0.0,
            "grounding_observation_count": 0,
            "catalog_validity_rate": 0.0,
            "parse_success_rate": 0.0,
        }
    valid_rows = 0
    hallucinated_rows = 0
    grounding_rows = 0
    grounding_success = 0
    catalog_valid = 0
    parse_rows = 0
    parse_success = 0
    for row in predictions:
        candidates = set(row.get("candidate_items", []))
        predicted = dedupe(row.get("predicted_items", []))
        invalid = [item for item in predicted if item not in candidates]
        if predicted and not invalid:
            valid_rows += 1
        if invalid:
            hallucinated_rows += 1
        metadata = row.get("metadata") or {}
        if "grounding_success" in metadata:
            grounding_rows += 1
            grounding_success += int(bool(metadata.get("grounding_success")))
        if "is_catalog_valid" in metadata:
            catalog_valid += int(bool(metadata.get("is_catalog_valid")))
        if "parse_success" in metadata:
            parse_rows += 1
            parse_success += int(bool(metadata.get("parse_success")))
    total = len(predictions)
    return {
        "validity_rate": valid_rows / total,
        "hallucination_rate": hallucinated_rows / total,
        "valid_prediction_count": valid_rows,
        "hallucinated_prediction_count": hallucinated_rows,
        "grounding_success_rate": grounding_success / grounding_rows if grounding_rows else 0.0,
        "grounding_observation_count": grounding_rows,
        "catalog_validity_rate": catalog_valid / grounding_rows if grounding_rows else 0.0,
        "parse_success_rate": parse_success / parse_rows if parse_rows else 0.0,
    }

"""Prediction JSONL schema validation."""

from __future__ import annotations

from typing import Any

REQUIRED_FIELDS = {
    "user_id",
    "target_item",
    "candidate_items",
    "predicted_items",
    "method",
    "domain",
    "metadata",
}


def validate_prediction(record: dict[str, Any], *, row_number: int | None = None) -> dict[str, Any]:
    where = f"row {row_number}: " if row_number is not None else ""
    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        raise ValueError(f"{where}prediction missing required fields: {missing}")
    if not isinstance(record["user_id"], str) or not record["user_id"]:
        raise ValueError(f"{where}user_id must be a non-empty string")
    if not isinstance(record["target_item"], str) or not record["target_item"]:
        raise ValueError(f"{where}target_item must be a non-empty string")
    if not _is_str_list(record["candidate_items"]):
        raise ValueError(f"{where}candidate_items must be list[str]")
    if not _is_str_list(record["predicted_items"]):
        raise ValueError(f"{where}predicted_items must be list[str]")
    scores = record.get("scores")
    if scores in (None, []):
        scores = None
    elif not isinstance(scores, list):
        raise ValueError(f"{where}scores must be list[number], null, or empty")
    elif len(scores) != len(record["predicted_items"]):
        raise ValueError(f"{where}scores length must equal predicted_items length")
    else:
        for index, score in enumerate(scores):
            if not isinstance(score, (int, float)):
                raise ValueError(f"{where}scores[{index}] must be numeric")
    if not isinstance(record["method"], str) or not record["method"]:
        raise ValueError(f"{where}method must be a non-empty string")
    if not isinstance(record["domain"], str) or not record["domain"]:
        raise ValueError(f"{where}domain must be a non-empty string")
    if not isinstance(record["metadata"], dict):
        raise ValueError(f"{where}metadata must be an object")
    normalized = dict(record)
    normalized["scores"] = scores
    return normalized


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)

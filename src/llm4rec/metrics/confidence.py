"""Confidence diagnostics for generative recommendation observations."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any

CORRECTNESS_TARGET = "metadata.is_grounded_hit if present; otherwise top1_exact_match"


def confidence_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    rows, missing = _confidence_rows(predictions)
    if not rows:
        return {
            "correctness_target": CORRECTNESS_TARGET,
            "confidence_count": 0,
            "count_missing_confidence": missing,
            "mean_confidence": None,
            "confidence_correct_mean": None,
            "confidence_incorrect_mean": None,
            "confidence_accuracy_gap": None,
            "confidence_by_correctness": {},
            "selective_risk_coverage_data": [],
        }
    correct = [row["confidence"] for row in rows if row["correct"]]
    incorrect = [row["confidence"] for row in rows if not row["correct"]]
    correct_mean = mean(correct) if correct else None
    incorrect_mean = mean(incorrect) if incorrect else None
    return {
        "correctness_target": CORRECTNESS_TARGET,
        "confidence_count": len(rows),
        "count_missing_confidence": missing,
        "mean_confidence": mean(row["confidence"] for row in rows),
        "confidence_correct_mean": correct_mean,
        "confidence_incorrect_mean": incorrect_mean,
        "confidence_accuracy_gap": (
            correct_mean - incorrect_mean
            if correct_mean is not None and incorrect_mean is not None
            else None
        ),
        "confidence_by_correctness": {
            "correct": {"count": len(correct), "mean": correct_mean},
            "incorrect": {"count": len(incorrect), "mean": incorrect_mean},
        },
        "selective_risk_coverage_data": selective_risk_coverage_data(predictions),
    }


def selective_risk_coverage_data(predictions: list[dict[str, Any]]) -> list[dict[str, float]]:
    rows, _missing = _confidence_rows(predictions)
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: row["confidence"], reverse=True)
    output = []
    errors = 0
    for index, row in enumerate(ordered, start=1):
        if not row["correct"]:
            errors += 1
        output.append(
            {
                "coverage": index / len(ordered),
                "risk": errors / index,
                "threshold": row["confidence"],
            }
        )
    return output


def _confidence_rows(predictions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    rows = []
    missing = 0
    for record in predictions:
        metadata = record.get("metadata") or {}
        if "confidence" not in metadata or metadata.get("confidence") is None:
            missing += 1
            continue
        confidence = validate_confidence_value(metadata.get("confidence"))
        rows.append({"confidence": float(confidence), "correct": _is_correct(record)})
    return rows, missing


def validate_confidence_value(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence must be a numeric value in [0, 1]")
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be a numeric value in [0, 1]")
    return confidence


def _is_correct(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata") or {}
    if isinstance(metadata.get("is_grounded_hit"), bool):
        return bool(metadata["is_grounded_hit"])
    predicted = [str(item) for item in record.get("predicted_items", [])]
    return bool(predicted and predicted[0] == str(record.get("target_item")))

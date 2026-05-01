from __future__ import annotations

import pytest

from llm4rec.metrics.calibration import (
    brier_score,
    calibration_metrics,
    confidence_bucket_stats,
    expected_calibration_error,
)


ROWS = [
    {"confidence": 0.9, "correct": True},
    {"confidence": 0.1, "correct": False},
]


def test_calibration_metrics_basic_values() -> None:
    assert brier_score(ROWS) == pytest.approx(0.01)
    assert expected_calibration_error(ROWS, bins=2) == pytest.approx(0.1)
    stats = confidence_bucket_stats(ROWS, bins=2)
    assert stats[0]["count"] == 1
    assert stats[1]["count"] == 1


def test_calibration_metrics_extracts_from_predictions() -> None:
    metrics = calibration_metrics([
        {"target_item": "i1", "predicted_items": ["i1"], "metadata": {"confidence": 0.9}},
        {"target_item": "i2", "predicted_items": ["i1"], "metadata": {"confidence": 0.1}},
    ], bins=2)
    assert metrics["correctness_target"] == "metadata.is_grounded_hit if present; otherwise top1_exact_match"
    assert metrics["count_missing_confidence"] == 0
    assert metrics["ece"] == pytest.approx(0.1)


def test_calibration_boundaries_and_invalid_values() -> None:
    rows = [
        {"confidence": 0.0, "correct": False},
        {"confidence": 1.0, "correct": True},
    ]
    stats = confidence_bucket_stats(rows, bins=2)
    assert stats[0]["count"] == 1
    assert stats[0]["mean_confidence"] == 0.0
    assert stats[1]["count"] == 1
    assert stats[1]["mean_confidence"] == 1.0
    assert expected_calibration_error(rows, bins=2) == 0.0
    assert brier_score(rows) == 0.0
    with pytest.raises(ValueError, match="confidence must be"):
        confidence_bucket_stats([{"confidence": -0.01, "correct": True}], bins=2)

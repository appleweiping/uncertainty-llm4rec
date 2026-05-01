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
    assert metrics["count_missing_confidence"] == 0
    assert metrics["ece"] == pytest.approx(0.1)

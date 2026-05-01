from __future__ import annotations

import pytest

from llm4rec.metrics.confidence import confidence_metrics, selective_risk_coverage_data


def _predictions() -> list[dict[str, object]]:
    return [
        {
            "target_item": "i1",
            "predicted_items": ["i1"],
            "metadata": {"confidence": 0.8, "is_grounded_hit": True},
        },
        {
            "target_item": "i2",
            "predicted_items": ["i1"],
            "metadata": {"confidence": 0.6, "is_grounded_hit": False},
        },
        {"target_item": "i3", "predicted_items": ["i3"], "metadata": {}},
    ]


def test_confidence_metrics_split_correctness() -> None:
    metrics = confidence_metrics(_predictions())
    assert metrics["correctness_target"] == "metadata.is_grounded_hit if present; otherwise top1_exact_match"
    assert metrics["confidence_count"] == 2
    assert metrics["count_missing_confidence"] == 1
    assert metrics["confidence_correct_mean"] == 0.8
    assert metrics["confidence_incorrect_mean"] == 0.6
    assert metrics["confidence_accuracy_gap"] == 0.20000000000000007


def test_selective_risk_curve_orders_by_confidence() -> None:
    curve = selective_risk_coverage_data(_predictions())
    assert curve[0]["coverage"] == 0.5
    assert curve[-1]["risk"] == 0.5


def test_confidence_metrics_accepts_zero_and_one_boundaries() -> None:
    metrics = confidence_metrics([
        {"target_item": "i1", "predicted_items": ["i2"], "metadata": {"confidence": 0.0}},
        {"target_item": "i2", "predicted_items": ["i2"], "metadata": {"confidence": 1.0}},
    ])
    assert metrics["confidence_count"] == 2
    assert metrics["mean_confidence"] == 0.5


def test_confidence_metrics_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence must be"):
        confidence_metrics([{"target_item": "i1", "predicted_items": ["i1"], "metadata": {"confidence": "0.9"}}])
    with pytest.raises(ValueError, match="confidence must be"):
        confidence_metrics([{"target_item": "i1", "predicted_items": ["i1"], "metadata": {"confidence": 1.01}}])

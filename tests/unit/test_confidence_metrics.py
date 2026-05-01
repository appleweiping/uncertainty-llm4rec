from __future__ import annotations

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
    assert metrics["confidence_count"] == 2
    assert metrics["count_missing_confidence"] == 1
    assert metrics["confidence_correct_mean"] == 0.8
    assert metrics["confidence_incorrect_mean"] == 0.6
    assert metrics["confidence_accuracy_gap"] == 0.20000000000000007


def test_selective_risk_curve_orders_by_confidence() -> None:
    curve = selective_risk_coverage_data(_predictions())
    assert curve[0]["coverage"] == 0.5
    assert curve[-1]["risk"] == 0.5

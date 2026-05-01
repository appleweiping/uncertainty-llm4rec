from __future__ import annotations

from llm4rec.metrics.validity import validity_metrics


def test_validity_metrics_valid_invalid_empty_and_duplicate() -> None:
    rows = [
        {"candidate_items": ["i1", "i2"], "predicted_items": ["i1", "i1"]},
        {"candidate_items": ["i1", "i2"], "predicted_items": ["ghost"]},
        {"candidate_items": ["i1", "i2"], "predicted_items": []},
    ]
    metrics = validity_metrics(rows)
    assert metrics["valid_prediction_count"] == 1
    assert metrics["hallucinated_prediction_count"] == 1
    assert metrics["validity_rate"] == 1 / 3
    assert metrics["hallucination_rate"] == 1 / 3


def test_validity_metrics_empty_input() -> None:
    metrics = validity_metrics([])
    assert metrics["validity_rate"] == 0.0
    assert metrics["hallucination_rate"] == 0.0

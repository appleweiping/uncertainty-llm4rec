from __future__ import annotations

from llm4rec.metrics.coverage import catalog_coverage_at_k, item_coverage_at_k, user_coverage


def test_item_and_catalog_coverage_at_k() -> None:
    predictions = [
        {"candidate_items": ["i1", "i2", "i3"], "predicted_items": ["i1", "i2"]},
        {"candidate_items": ["i1", "i2", "i3"], "predicted_items": ["i2", "i3"]},
    ]
    assert item_coverage_at_k(predictions, k=1)["coverage_rate"] == 2 / 3
    assert catalog_coverage_at_k(predictions, catalog_items=["i1", "i2", "i3", "i4"], k=2)["coverage_rate"] == 3 / 4


def test_user_coverage_counts_non_empty_recommendations() -> None:
    metrics = user_coverage([
        {"user_id": "u1", "predicted_items": ["i1"]},
        {"user_id": "u2", "predicted_items": []},
    ])
    assert metrics["user_coverage_rate"] == 0.5
    assert metrics["unique_users"] == 2

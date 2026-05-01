from __future__ import annotations

from llm4rec.metrics.slicing import slice_predictions


def test_slice_predictions_supports_nested_metadata_keys() -> None:
    grouped = slice_predictions(
        [
            {"metadata": {"user_history_bucket": "cold_user"}},
            {"metadata": {"user_history_bucket": "warm_user"}},
            {"metadata": {}},
        ],
        key="metadata.user_history_bucket",
    )
    assert sorted(grouped) == ["cold_user", "unknown", "warm_user"]
    assert len(grouped["unknown"]) == 1

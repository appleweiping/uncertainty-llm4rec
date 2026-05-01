from __future__ import annotations

from llm4rec.llm.cost_tracker import CostTracker
from llm4rec.metrics.efficiency import efficiency_metrics


def test_cost_tracker_accumulates_usage_and_latency() -> None:
    tracker = CostTracker(prompt_token_cost_per_1k=0.1, completion_token_cost_per_1k=0.2)
    tracker.add(usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, latency_seconds=0.2)
    tracker.add(usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}, latency_seconds=0.4, cache_hit=True)
    summary = tracker.summary()
    assert summary["total_tokens"] == 45
    assert summary["cache_hit_rate"] == 0.5
    assert summary["estimated_cost"] > 0


def test_efficiency_metrics_reads_prediction_metadata() -> None:
    metrics = efficiency_metrics([
        {"metadata": {"latency_seconds": 0.1, "token_usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}, "cache_hit": False}},
        {"metadata": {"latency_seconds": 0.3, "token_usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}, "cache_hit": True}},
    ])
    assert metrics["total_tokens"] == 8
    assert metrics["cache_hit_rate"] == 0.5

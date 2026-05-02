from __future__ import annotations

from llm4rec.llm.cost_tracker import CostTracker
from llm4rec.metrics.efficiency import efficiency_metrics


def test_cost_tracker_accumulates_usage_and_latency() -> None:
    tracker = CostTracker(prompt_token_cost_per_1k=0.1, completion_token_cost_per_1k=0.2)
    tracker.add(usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, latency_seconds=0.2)
    tracker.add(
        usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        latency_seconds=0.01,
        cache_hit=True,
        original_latency_seconds=0.4,
        original_estimated_cost_usd=0.004,
        replay_latency_seconds=0.01,
        replay_estimated_cost_usd=0.0,
    )
    summary = tracker.summary()
    assert summary["total_tokens"] == 45
    assert summary["cache_hit_rate"] == 0.5
    assert summary["live_cost_usd"] > 0
    assert summary["replay_cost_usd"] == 0.0
    assert summary["original_cached_cost_usd"] == 0.004
    assert summary["effective_cost_usd"] == summary["live_cost_usd"] + 0.004
    assert summary["replay_latency_seconds_sum"] == 0.01
    assert summary["original_cached_latency_seconds_sum"] == 0.4


def test_efficiency_metrics_reads_prediction_metadata() -> None:
    metrics = efficiency_metrics([
        {"metadata": {"latency_seconds": 0.1, "token_usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}, "cache_hit": False}},
        {
            "metadata": {
                "latency_seconds": 0.003,
                "token_usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                "cache_hit": True,
                "provider_metadata": {
                    "original_latency_seconds": 0.3,
                    "original_estimated_cost_usd": 0.002,
                    "replay_latency_seconds": 0.003,
                    "replay_estimated_cost_usd": 0.0,
                },
            }
        },
    ])
    assert metrics["total_tokens"] == 8
    assert metrics["cache_hit_rate"] == 0.5
    assert metrics["live_provider_requests"] == 1
    assert metrics["cache_hit_requests"] == 1
    assert metrics["original_cached_total_tokens"] == 5
    assert metrics["original_cached_cost_usd"] == 0.002
    assert metrics["replay_latency_seconds_sum"] == 0.003
    assert metrics["original_cached_latency_seconds_sum"] == 0.3

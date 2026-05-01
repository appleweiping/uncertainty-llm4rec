"""Efficiency metrics for LLM provider metadata."""

from __future__ import annotations

from statistics import mean, median
from typing import Any


def efficiency_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    estimated_cost = 0.0
    cache_hits = 0
    counted = 0
    for row in predictions:
        metadata = row.get("metadata") or {}
        if isinstance(metadata.get("latency_seconds"), (int, float)):
            latencies.append(float(metadata["latency_seconds"]))
        usage = metadata.get("token_usage") or {}
        if isinstance(usage, dict):
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
            total_tokens += int(usage.get("total_tokens") or 0)
        estimated_cost += float(metadata.get("estimated_cost") or 0.0)
        if "cache_hit" in metadata:
            counted += 1
            cache_hits += int(bool(metadata["cache_hit"]))
    return {
        "latency_mean": mean(latencies) if latencies else 0.0,
        "latency_p50": median(latencies) if latencies else 0.0,
        "latency_p95": _percentile(latencies, 0.95) if latencies else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost": estimated_cost,
        "cache_hit_rate": cache_hits / counted if counted else 0.0,
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]

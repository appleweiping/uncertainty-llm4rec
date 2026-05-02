"""Efficiency metrics for LLM provider metadata."""

from __future__ import annotations

from statistics import mean, median
from typing import Any


def efficiency_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = []
    original_live_latencies = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    live_prompt_tokens = 0
    live_completion_tokens = 0
    live_total_tokens = 0
    original_cached_prompt_tokens = 0
    original_cached_completion_tokens = 0
    original_cached_total_tokens = 0
    live_cost_usd = 0.0
    replay_cost_usd = 0.0
    original_cached_cost_usd = 0.0
    live_latency_seconds_sum = 0.0
    replay_latency_seconds_sum = 0.0
    original_cached_latency_seconds_sum = 0.0
    cache_hits = 0
    counted = 0
    for row in predictions:
        metadata = row.get("metadata") or {}
        provider_metadata = metadata.get("provider_metadata") or {}
        if not isinstance(provider_metadata, dict):
            provider_metadata = {}
        latency_seconds = _number(metadata.get("latency_seconds"))
        if latency_seconds is not None:
            latencies.append(latency_seconds)
        usage = metadata.get("token_usage") or {}
        if isinstance(usage, dict):
            row_prompt_tokens = int(usage.get("prompt_tokens") or 0)
            row_completion_tokens = int(usage.get("completion_tokens") or 0)
            row_total_tokens = int(usage.get("total_tokens") or row_prompt_tokens + row_completion_tokens)
        else:
            row_prompt_tokens = 0
            row_completion_tokens = 0
            row_total_tokens = 0
        prompt_tokens += row_prompt_tokens
        completion_tokens += row_completion_tokens
        total_tokens += row_total_tokens
        if "cache_hit" in metadata:
            counted += 1
            cache_hit = bool(metadata["cache_hit"])
            cache_hits += int(cache_hit)
        else:
            cache_hit = False
        replay_cost = _number(metadata.get("replay_estimated_cost_usd"))
        if replay_cost is None:
            replay_cost = _number(provider_metadata.get("replay_estimated_cost_usd"))
        original_cached_cost = _number(metadata.get("original_estimated_cost_usd"))
        if original_cached_cost is None:
            original_cached_cost = _number(provider_metadata.get("original_estimated_cost_usd"))
        if original_cached_cost is None:
            original_cached_cost = 0.0
        live_cost = _number(metadata.get("estimated_cost")) or 0.0
        if cache_hit:
            replay_cost_usd += replay_cost or 0.0
            original_cached_cost_usd += original_cached_cost
            original_cached_prompt_tokens += row_prompt_tokens
            original_cached_completion_tokens += row_completion_tokens
            original_cached_total_tokens += row_total_tokens
            replay_latency = _number(metadata.get("replay_latency_seconds"))
            if replay_latency is None:
                replay_latency = _number(provider_metadata.get("replay_latency_seconds"))
            if replay_latency is None:
                replay_latency = latency_seconds or 0.0
            replay_latency_seconds_sum += replay_latency
            original_latency = _number(metadata.get("original_latency_seconds"))
            if original_latency is None:
                original_latency = _number(provider_metadata.get("original_latency_seconds"))
            if original_latency is None:
                original_latency = 0.0
            original_cached_latency_seconds_sum += original_latency
            original_live_latencies.append(original_latency)
        else:
            live_cost_usd += live_cost
            live_prompt_tokens += row_prompt_tokens
            live_completion_tokens += row_completion_tokens
            live_total_tokens += row_total_tokens
            live_latency = latency_seconds or 0.0
            live_latency_seconds_sum += live_latency
            original_live_latencies.append(live_latency)
    effective_cost_usd = live_cost_usd + original_cached_cost_usd
    return {
        "latency_mean": mean(latencies) if latencies else 0.0,
        "latency_p50": median(latencies) if latencies else 0.0,
        "latency_p95": _percentile(latencies, 0.95) if latencies else 0.0,
        "latency_p50_seconds": median(latencies) if latencies else 0.0,
        "latency_p95_seconds": _percentile(latencies, 0.95) if latencies else 0.0,
        "original_live_latency_p50_seconds": (
            median(original_live_latencies) if original_live_latencies else 0.0
        ),
        "original_live_latency_p95_seconds": (
            _percentile(original_live_latencies, 0.95) if original_live_latencies else 0.0
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "live_prompt_tokens": live_prompt_tokens,
        "live_completion_tokens": live_completion_tokens,
        "live_total_tokens": live_total_tokens,
        "original_cached_prompt_tokens": original_cached_prompt_tokens,
        "original_cached_completion_tokens": original_cached_completion_tokens,
        "original_cached_total_tokens": original_cached_total_tokens,
        "live_cost_usd": live_cost_usd,
        "replay_cost_usd": replay_cost_usd,
        "original_cached_cost_usd": original_cached_cost_usd,
        "effective_cost_usd": effective_cost_usd,
        "estimated_cost": effective_cost_usd,
        "live_latency_seconds_sum": live_latency_seconds_sum,
        "replay_latency_seconds_sum": replay_latency_seconds_sum,
        "original_cached_latency_seconds_sum": original_cached_latency_seconds_sum,
        "cache_hit_rate": cache_hits / counted if counted else 0.0,
        "total_requests": counted,
        "live_provider_requests": counted - cache_hits,
        "cache_hit_requests": cache_hits,
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None

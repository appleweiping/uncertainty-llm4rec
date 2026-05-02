"""Token, latency, and cost aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any


@dataclass(slots=True)
class CostTracker:
    prompt_token_cost_per_1k: float = 0.0
    completion_token_cost_per_1k: float = 0.0
    records: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        *,
        usage: dict[str, int] | None = None,
        latency_seconds: float | None = None,
        cache_hit: bool = False,
        original_usage: dict[str, int] | None = None,
        original_latency_seconds: float | None = None,
        original_estimated_cost_usd: float | None = None,
        replay_latency_seconds: float | None = None,
        replay_estimated_cost_usd: float = 0.0,
    ) -> dict[str, Any]:
        usage = usage or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
        estimated_cost = (
            prompt_tokens / 1000.0 * self.prompt_token_cost_per_1k
            + completion_tokens / 1000.0 * self.completion_token_cost_per_1k
        )
        record = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": float(latency_seconds or 0.0),
            "cache_hit": bool(cache_hit),
            "estimated_cost": estimated_cost,
            "live_cost_usd": 0.0 if cache_hit else estimated_cost,
            "replay_cost_usd": float(replay_estimated_cost_usd or 0.0) if cache_hit else 0.0,
            "original_cached_cost_usd": (
                float(original_estimated_cost_usd or 0.0) if cache_hit else 0.0
            ),
            "effective_cost_usd": (
                float(original_estimated_cost_usd or 0.0) if cache_hit else estimated_cost
            ),
            "original_usage": dict(original_usage or usage),
            "original_latency_seconds": float(original_latency_seconds or 0.0) if cache_hit else float(latency_seconds or 0.0),
            "replay_latency_seconds": float(replay_latency_seconds or latency_seconds or 0.0) if cache_hit else 0.0,
        }
        self.records.append(record)
        return record

    def summary(self) -> dict[str, Any]:
        latencies = [float(row["latency_seconds"]) for row in self.records]
        original_live_latencies = [
            float(row.get("original_latency_seconds") or row.get("latency_seconds") or 0.0)
            for row in self.records
        ]
        cache_records = [row for row in self.records if row["cache_hit"]]
        live_records = [row for row in self.records if not row["cache_hit"]]
        live_cost_usd = sum(float(row.get("live_cost_usd") or row.get("estimated_cost") or 0.0) for row in live_records)
        original_cached_cost_usd = sum(float(row.get("original_cached_cost_usd") or 0.0) for row in cache_records)
        effective_cost_usd = live_cost_usd + original_cached_cost_usd
        return {
            "request_count": len(self.records),
            "total_requests": len(self.records),
            "live_provider_requests": len(live_records),
            "cache_hit_requests": len(cache_records),
            "prompt_tokens": sum(int(row["prompt_tokens"]) for row in self.records),
            "completion_tokens": sum(int(row["completion_tokens"]) for row in self.records),
            "total_tokens": sum(int(row["total_tokens"]) for row in self.records),
            "live_cost_usd": live_cost_usd,
            "replay_cost_usd": sum(float(row.get("replay_cost_usd") or 0.0) for row in cache_records),
            "original_cached_cost_usd": original_cached_cost_usd,
            "effective_cost_usd": effective_cost_usd,
            "estimated_cost": effective_cost_usd,
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
            "live_latency_seconds_sum": sum(float(row["latency_seconds"]) for row in live_records),
            "replay_latency_seconds_sum": sum(float(row.get("replay_latency_seconds") or 0.0) for row in cache_records),
            "original_cached_latency_seconds_sum": sum(
                float(row.get("original_latency_seconds") or 0.0) for row in cache_records
            ),
            "cache_hit_rate": (
                sum(1 for row in self.records if row["cache_hit"]) / len(self.records)
                if self.records else 0.0
            ),
        }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]

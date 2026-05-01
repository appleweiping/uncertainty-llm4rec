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
        }
        self.records.append(record)
        return record

    def summary(self) -> dict[str, Any]:
        latencies = [float(row["latency_seconds"]) for row in self.records]
        return {
            "request_count": len(self.records),
            "prompt_tokens": sum(int(row["prompt_tokens"]) for row in self.records),
            "completion_tokens": sum(int(row["completion_tokens"]) for row in self.records),
            "total_tokens": sum(int(row["total_tokens"]) for row in self.records),
            "estimated_cost": sum(float(row["estimated_cost"]) for row in self.records),
            "latency_mean": mean(latencies) if latencies else 0.0,
            "latency_p50": median(latencies) if latencies else 0.0,
            "latency_p95": _percentile(latencies, 0.95) if latencies else 0.0,
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

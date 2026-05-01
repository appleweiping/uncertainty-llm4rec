"""Aggregate metrics across run directories."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


def aggregate_run_metrics(input_dir: str | Path, *, output_dir: str | Path) -> dict[str, Any]:
    runs = load_run_metrics(input_dir)
    rows = aggregate_metric_rows(runs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "aggregate_metrics.csv"
    _write_csv(csv_path, rows)
    return {
        "run_count": len(runs),
        "row_count": len(rows),
        "aggregate_metrics_csv": str(csv_path),
        "rows": rows,
    }


def load_run_metrics(input_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(input_dir)
    paths = sorted(root.glob("*/metrics.json")) if root.is_dir() else [root]
    runs = []
    for path in paths:
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        metadata = dict(metrics.get("metadata") or {})
        metadata.setdefault("run_id", path.parent.name)
        metadata.setdefault("run_dir", str(path.parent))
        runs.append({"path": str(path), "metrics": metrics, "metadata": metadata})
    return runs


def aggregate_metric_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for run in runs:
        metrics = run["metrics"]
        metadata = run["metadata"]
        method = str(metadata.get("method") or _single_or_unknown(metrics.get("metadata", {}).get("methods")) or "unknown")
        dataset = str(metadata.get("dataset") or metrics.get("metadata", {}).get("dataset") or "unknown")
        for domain, values in _domain_metric_maps(metrics):
            for metric_name, value in _flatten_numeric(values):
                grouped[(method, dataset, domain, metric_name)].append(float(value))
    rows = []
    for (method, dataset, domain, metric_name), values in sorted(grouped.items()):
        count = len(values)
        std = stdev(values) if count > 1 else 0.0
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "domain": domain,
                "metric": metric_name,
                "mean": mean(values) if values else 0.0,
                "std": std,
                "count": count,
                "ci95": 1.96 * std / math.sqrt(count) if count > 1 else 0.0,
            }
        )
    return rows


def _domain_metric_maps(metrics: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(metrics.get("aggregate"), dict):
        yield "aggregate", metrics["aggregate"]
    domains = metrics.get("by_domain") or metrics.get("per_domain") or {}
    if isinstance(domains, dict):
        for domain, values in sorted(domains.items()):
            if isinstance(values, dict):
                yield str(domain), values


def _flatten_numeric(value: Any, prefix: str = "") -> Iterable[tuple[str, float]]:
    if isinstance(value, dict):
        for key, item in sorted(value.items()):
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_numeric(item, name)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        yield prefix, float(value)


def _single_or_unknown(values: Any) -> str | None:
    if isinstance(values, list) and len(values) == 1:
        return str(values[0])
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "dataset", "domain", "metric", "mean", "std", "count", "ci95"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

"""Evaluation artifact export wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.io.artifacts import write_json, write_metrics_csv


def export_metrics(metrics: dict[str, Any], *, output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    json_path = output / "metrics.json"
    csv_path = output / "metrics.csv"
    write_json(json_path, metrics)
    write_metrics_csv(csv_path, metrics)
    return {"metrics_json": str(json_path), "metrics_csv": str(csv_path)}

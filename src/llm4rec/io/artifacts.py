"""Run artifact creation and export helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from llm4rec.utils.env import collect_environment


def create_run_dir(output_dir: str | Path, run_id: str) -> Path:
    run_dir = Path(output_dir) / run_id
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_environment(run_dir: str | Path) -> Path:
    path = Path(run_dir) / "environment.json"
    write_json(path, collect_environment())
    return path


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_metrics_csv(path: str | Path, metrics: dict[str, Any]) -> None:
    rows = list(_flatten_metrics(metrics))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> Iterable[dict[str, Any]]:
    for key, value in sorted(metrics.items()):
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            yield from _flatten_metrics(value, prefix=name)
        elif isinstance(value, list):
            yield {"scope": "aggregate", "metric": name, "value": json.dumps(value, sort_keys=True)}
        else:
            yield {"scope": "aggregate", "metric": name, "value": value}

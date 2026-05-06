#!/usr/bin/env python3
"""Summarize controlled-baseline training, scoring, import, and metric status."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_NAMES = [
    "tallrec_qwen3_lora_amazon_beauty",
    "openp5_style_qwen3_lora_amazon_beauty",
    "dealrec_qwen3_lora_amazon_beauty",
    "lc_rec_qwen3_lora_amazon_beauty",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="*", default=DEFAULT_NAMES)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "summary" / "controlled_baselines")
    args = parser.parse_args()
    rows = [summarize_name(name) for name in args.names]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "controlled_baseline_status.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    with (args.output_dir / "controlled_baseline_status.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "name",
            "project",
            "training_status",
            "score_rows",
            "candidate_score_lines",
            "adapter_exists",
            "run_exists",
            "prediction_lines",
            "metric_count",
            "recall@10",
            "ndcg@10",
            "mrr@10",
            "next_action",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def summarize_name(name: str) -> dict[str, Any]:
    base = ROOT / "outputs" / "server_training" / "controlled_baselines" / name
    run = ROOT / "outputs" / "runs" / f"{name}_seed13"
    manifest = _read_json(base / "controlled_baseline_manifest.json")
    summary = _read_json(base / "training_scoring_summary.json")
    metrics = _read_json(run / "metrics.json")
    score_path = base / "candidate_scores.csv"
    prediction_path = run / "predictions.jsonl"
    metric_agg = metrics.get("aggregate", {}) if isinstance(metrics, dict) else {}
    row = {
        "name": name,
        "project": manifest.get("project", ""),
        "training_status": summary.get("status", "missing"),
        "score_rows": summary.get("score_rows", 0),
        "candidate_score_lines": _line_count(score_path),
        "adapter_exists": (base / "adapter").exists(),
        "run_exists": run.exists(),
        "prediction_lines": _line_count(prediction_path),
        "metric_count": metrics.get("count", 0) if isinstance(metrics, dict) else 0,
        "recall@10": metric_agg.get("recall@10", ""),
        "ndcg@10": metric_agg.get("ndcg@10", ""),
        "mrr@10": metric_agg.get("mrr@10", ""),
        "next_action": _next_action(name, summary=summary, metrics=metrics),
    }
    return row


def _next_action(name: str, *, summary: dict[str, Any], metrics: dict[str, Any]) -> str:
    if not summary:
        return "run_training_scoring"
    if name.startswith("openp5_style") and int(summary.get("score_rows") or 0) <= 2:
        return "optimize_scoring_before_full_run"
    if not metrics:
        return "import_and_evaluate"
    return "complete_or_ready_for_analysis"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


if __name__ == "__main__":
    raise SystemExit(main())

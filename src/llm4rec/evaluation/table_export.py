"""Paper-ready table export helpers without prose claims."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from llm4rec.evaluation.aggregation import aggregate_metric_rows, load_run_metrics


def export_phase5_tables(input_dir: str | Path, *, output_dir: str | Path) -> dict[str, str]:
    runs = load_run_metrics(input_dir)
    rows = aggregate_metric_rows(runs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "aggregate_metrics.csv"
    md_path = output / "aggregate_metrics.md"
    tex_path = output / "aggregate_metrics.tex"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    _write_latex(tex_path, rows)
    exported = {
        "aggregate_metrics_csv": str(csv_path),
        "aggregate_metrics_md": str(md_path),
        "aggregate_metrics_tex": str(tex_path),
    }
    reliability = _collect_reliability_rows(runs)
    if reliability:
        path = output / "reliability_diagram.csv"
        _write_csv(path, reliability)
        exported["reliability_diagram_csv"] = str(path)
    risk = _collect_risk_coverage_rows(runs)
    if risk:
        path = output / "risk_coverage.csv"
        _write_csv(path, risk)
        exported["risk_coverage_csv"] = str(path)
    confidence_popularity = _collect_confidence_popularity_rows(runs)
    if confidence_popularity:
        path = output / "confidence_by_popularity_bucket.csv"
        _write_csv(path, confidence_popularity)
        exported["confidence_by_popularity_bucket_csv"] = str(path)
    summary_path = output / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_count": len(runs),
                "table_row_count": len(rows),
                "is_experiment_result": False,
                "note": "Phase 5 smoke/export artifacts only; tables contain data and no paper claims.",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    exported["experiment_summary_json"] = str(summary_path)
    return exported


def _collect_reliability_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        prefix = _row_prefix(run)
        data = run["metrics"].get("aggregate", {}).get("calibration", {}).get("reliability_diagram_data", [])
        for bucket in data if isinstance(data, list) else []:
            rows.append({**prefix, **bucket})
    return rows


def _collect_risk_coverage_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        prefix = _row_prefix(run)
        data = run["metrics"].get("aggregate", {}).get("confidence", {}).get("selective_risk_coverage_data", [])
        for point in data if isinstance(data, list) else []:
            rows.append({**prefix, **point})
    return rows


def _collect_confidence_popularity_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        prefix = _row_prefix(run)
        data = run["metrics"].get("aggregate", {}).get("long_tail", {}).get("confidence_by_popularity_bucket", {})
        if not isinstance(data, dict):
            continue
        for bucket, stats in sorted(data.items()):
            if isinstance(stats, dict):
                rows.append({**prefix, "bucket": bucket, **stats})
    return rows


def _row_prefix(run: dict[str, Any]) -> dict[str, Any]:
    metadata = run.get("metadata") or run.get("metrics", {}).get("metadata", {})
    return {
        "run_id": str(metadata.get("run_id") or Path(str(run.get("path", ""))).parent.name),
        "method": str(metadata.get("method") or "unknown"),
        "dataset": str(metadata.get("dataset") or "unknown"),
        "seed": metadata.get("seed", "unknown"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["method", "dataset", "domain", "metric", "mean", "std", "count", "ci95"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["method", "dataset", "domain", "metric", "mean", "std", "count", "ci95"]
    lines = [
        "\\begin{tabular}{llllllll}",
        "\\toprule",
        " & ".join(fields) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_escape_latex(_format_cell(row.get(field, ""))) for field in fields) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _escape_latex(value: str) -> str:
    return value.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")

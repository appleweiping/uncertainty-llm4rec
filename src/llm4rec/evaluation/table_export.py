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
    candidate_sensitivity = _collect_candidate_sensitivity_rows(runs)
    if candidate_sensitivity:
        csv_path = output / "candidate_sensitivity.csv"
        md_path = output / "candidate_sensitivity.md"
        tex_path = output / "candidate_sensitivity.tex"
        _write_csv(csv_path, candidate_sensitivity)
        _write_candidate_markdown(md_path, candidate_sensitivity)
        _write_candidate_latex(tex_path, candidate_sensitivity)
        exported["candidate_sensitivity_csv"] = str(csv_path)
        exported["candidate_sensitivity_md"] = str(md_path)
        exported["candidate_sensitivity_tex"] = str(tex_path)
    summary_path = output / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_count": len(runs),
                "table_row_count": len(rows),
                "is_experiment_result": False,
                "evidence_labels": _evidence_labels(runs),
                "mock_llm_notices": _mock_llm_notices(runs),
                "note": _summary_note(runs),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    exported["experiment_summary_json"] = str(summary_path)
    return exported


def _evidence_labels(runs: list[dict[str, Any]]) -> list[str]:
    labels = {
        str((run.get("metrics", {}).get("metadata") or {}).get("evidence_label") or "").strip()
        for run in runs
    }
    labels.discard("")
    return sorted(labels)


def _mock_llm_notices(runs: list[dict[str, Any]]) -> list[str]:
    notices = {
        str((run.get("metrics", {}).get("metadata") or {}).get("mock_llm_notice") or "").strip()
        for run in runs
    }
    notices.discard("")
    return sorted(notices)


def _summary_note(runs: list[dict[str, Any]]) -> str:
    labels = _evidence_labels(runs)
    if labels:
        return "Tables contain data only; evidence labels are preserved from run metrics and no paper claims are made."
    return "Phase 5 smoke/export artifacts only; tables contain data and no paper claims."


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


def _collect_candidate_sensitivity_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        prefix = _row_prefix(run)
        candidate_size = prefix.get("candidate_size")
        if candidate_size in (None, "", "unknown"):
            continue
        metrics = run["metrics"]
        aggregate = metrics.get("aggregate", {}) if isinstance(metrics.get("aggregate"), dict) else {}
        efficiency = aggregate.get("efficiency", {}) if isinstance(aggregate.get("efficiency"), dict) else {}
        confidence = aggregate.get("confidence", {}) if isinstance(aggregate.get("confidence"), dict) else {}
        calibration = aggregate.get("calibration", {}) if isinstance(aggregate.get("calibration"), dict) else {}
        long_tail = aggregate.get("long_tail", {}) if isinstance(aggregate.get("long_tail"), dict) else {}
        novelty = aggregate.get("novelty", {}) if isinstance(aggregate.get("novelty"), dict) else {}
        coverage = aggregate.get("coverage_metrics", {}) if isinstance(aggregate.get("coverage_metrics"), dict) else {}
        predictions = _load_predictions_for_run(run)
        high_low = _high_low_confidence_counts(predictions)
        rows.append(
            {
                **prefix,
                "recall@10": aggregate.get("recall@10"),
                "ndcg@10": aggregate.get("ndcg@10"),
                "mrr@10": aggregate.get("mrr@10"),
                "validity_rate": aggregate.get("validity_rate"),
                "hallucination_rate": aggregate.get("hallucination_rate"),
                "parse_success_rate": aggregate.get("parse_success_rate"),
                "grounding_success_rate": aggregate.get("grounding_success_rate"),
                "mean_confidence": confidence.get("mean_confidence"),
                "ece": calibration.get("ece"),
                "brier": calibration.get("brier_score"),
                "high_confidence_wrong_count": high_low["high_confidence_wrong_count"],
                "low_confidence_correct_count": high_low["low_confidence_correct_count"],
                "coverage@10": (coverage.get("catalog_coverage@10") or {}).get("coverage_rate")
                if isinstance(coverage.get("catalog_coverage@10"), dict)
                else aggregate.get("coverage@10"),
                "novelty@10": novelty.get("novelty@10"),
                "head_recall@10": (long_tail.get("recall_by_popularity_bucket@10") or {}).get("head")
                if isinstance(long_tail.get("recall_by_popularity_bucket@10"), dict)
                else None,
                "mid_recall@10": (long_tail.get("recall_by_popularity_bucket@10") or {}).get("mid")
                if isinstance(long_tail.get("recall_by_popularity_bucket@10"), dict)
                else None,
                "tail_recall@10": (long_tail.get("recall_by_popularity_bucket@10") or {}).get("tail")
                if isinstance(long_tail.get("recall_by_popularity_bucket@10"), dict)
                else None,
                "head_hit_rate@10": (long_tail.get("hit_rate_by_popularity_bucket@10") or {}).get("head")
                if isinstance(long_tail.get("hit_rate_by_popularity_bucket@10"), dict)
                else None,
                "mid_hit_rate@10": (long_tail.get("hit_rate_by_popularity_bucket@10") or {}).get("mid")
                if isinstance(long_tail.get("hit_rate_by_popularity_bucket@10"), dict)
                else None,
                "tail_hit_rate@10": (long_tail.get("hit_rate_by_popularity_bucket@10") or {}).get("tail")
                if isinstance(long_tail.get("hit_rate_by_popularity_bucket@10"), dict)
                else None,
                "effective_cost_usd": efficiency.get("effective_cost_usd", efficiency.get("estimated_cost")),
                "live_cost_usd": efficiency.get("live_cost_usd"),
                "original_cached_cost_usd": efficiency.get("original_cached_cost_usd"),
                "total_tokens": efficiency.get("total_tokens"),
                "cache_hit_rate": efficiency.get("cache_hit_rate"),
                "latency_p50_seconds": efficiency.get("latency_p50_seconds", efficiency.get("latency_p50")),
                "latency_p95_seconds": efficiency.get("latency_p95_seconds", efficiency.get("latency_p95")),
                "original_live_latency_p50_seconds": efficiency.get("original_live_latency_p50_seconds"),
                "original_live_latency_p95_seconds": efficiency.get("original_live_latency_p95_seconds"),
            }
        )
    return sorted(rows, key=lambda row: (int(row.get("candidate_size") or 0), str(row.get("method") or "")))


def _load_predictions_for_run(run: dict[str, Any]) -> list[dict[str, Any]]:
    metrics_path = Path(str(run.get("path") or ""))
    predictions_path = metrics_path.parent / "predictions.jsonl"
    if not predictions_path.exists():
        return []
    rows = []
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _high_low_confidence_counts(predictions: list[dict[str, Any]]) -> dict[str, int]:
    high_wrong = 0
    low_correct = 0
    for row in predictions:
        metadata = row.get("metadata") or {}
        confidence = metadata.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            continue
        correct = _is_correct_at_10(row)
        if float(confidence) >= 0.8 and not correct:
            high_wrong += 1
        if float(confidence) < 0.5 and correct:
            low_correct += 1
    return {
        "high_confidence_wrong_count": high_wrong,
        "low_confidence_correct_count": low_correct,
    }


def _is_correct_at_10(row: dict[str, Any]) -> bool:
    target = str(row.get("target_item"))
    return target in [str(item) for item in row.get("predicted_items", [])[:10]]


def _row_prefix(run: dict[str, Any]) -> dict[str, Any]:
    metadata = run.get("metadata") or run.get("metrics", {}).get("metadata", {})
    return {
        "run_id": str(metadata.get("run_id") or Path(str(run.get("path", ""))).parent.name),
        "method": str(metadata.get("method") or "unknown"),
        "dataset": str(metadata.get("dataset") or "unknown"),
        "seed": metadata.get("seed", "unknown"),
        "candidate_size": metadata.get("candidate_size", "unknown"),
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


def _write_candidate_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "candidate_size",
        "method",
        "recall@10",
        "ndcg@10",
        "mrr@10",
        "validity_rate",
        "hallucination_rate",
        "parse_success_rate",
        "grounding_success_rate",
        "mean_confidence",
        "ece",
        "brier",
        "effective_cost_usd",
        "cache_hit_rate",
    ]
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


def _write_candidate_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["candidate_size", "method", "recall@10", "ndcg@10", "mrr@10", "validity_rate"]
    lines = [
        "\\begin{tabular}{llllll}",
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

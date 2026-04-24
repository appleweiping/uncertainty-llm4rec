from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


DEFAULT_SPECS: list[dict[str, str]] = [
    {
        "route_family": "local_v2_light",
        "domain": "beauty",
        "scale_tag": "full",
        "exp_name": "beauty_qwen3_local_replay_v2_pointwise_full",
    },
    {
        "route_family": "local_v2_light",
        "domain": "books",
        "scale_tag": "small3000",
        "exp_name": "books_qwen3_local_pointwise_small3000",
    },
    {
        "route_family": "local_v2_light",
        "domain": "electronics",
        "scale_tag": "small3000",
        "exp_name": "electronics_qwen3_local_pointwise_small3000",
    },
    {
        "route_family": "local_v2_light",
        "domain": "movies",
        "scale_tag": "small3000",
        "exp_name": "movies_qwen3_local_pointwise_small3000",
    },
    {
        "route_family": "historical_api_light",
        "domain": "beauty",
        "scale_tag": "full",
        "exp_name": "beauty_deepseek_pointwise_full",
    },
    {
        "route_family": "historical_api_light",
        "domain": "books",
        "scale_tag": "full3000",
        "exp_name": "books_deepseek_pointwise_full3000",
    },
    {
        "route_family": "historical_api_light",
        "domain": "electronics",
        "scale_tag": "full3000",
        "exp_name": "electronics_deepseek_pointwise_full3000",
    },
    {
        "route_family": "historical_api_light",
        "domain": "movies",
        "scale_tag": "full3000",
        "exp_name": "movies_deepseek_pointwise_full3000",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit old/light pointwise yes-no verbalized-confidence signals."
    )
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/summary/week7_9_light_pointwise_audit.csv",
    )
    parser.add_argument(
        "--max_raw_rows",
        type=int,
        default=0,
        help="Optional cap for raw prediction audit rows. 0 means read all rows.",
    )
    return parser.parse_args()


def _single_row_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def _multi_row_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _calibration_test_metrics(path: Path) -> dict[str, str]:
    rows = _multi_row_csv(path)
    metrics: dict[str, str] = {}
    for row in rows:
        if str(row.get("split", "")).strip().lower() != "test":
            continue
        metric = str(row.get("metric", "")).strip().lower()
        if not metric:
            continue
        metrics[f"{metric}_before"] = str(row.get("before", ""))
        metrics[f"{metric}_after"] = str(row.get("after", ""))
    return metrics


def _to_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | str:
    if not denominator:
        return ""
    return float(numerator) / float(denominator)


def _normalise_decision(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"yes", "y", "true", "1", "recommend"}:
        return "yes"
    if text in {"no", "n", "false", "0", "reject"}:
        return "no"
    return text or "missing"


def _prediction_paths(exp_dir: Path) -> list[Path]:
    predictions_dir = exp_dir / "predictions"
    preferred = [
        predictions_dir / "test_raw.jsonl",
        predictions_dir / "valid_raw.jsonl",
        predictions_dir / "rank_predictions.jsonl",
    ]
    paths = [path for path in preferred if path.exists()]
    if paths:
        return paths
    return sorted(predictions_dir.glob("*.jsonl")) if predictions_dir.exists() else []


def _audit_raw_predictions(exp_dir: Path, max_raw_rows: int) -> dict[str, Any]:
    paths = _prediction_paths(exp_dir)
    if not paths:
        return {"prediction_file_status": "missing"}

    confidences: list[float] = []
    labels: list[int] = []
    decisions: list[str] = []
    parse_success_count = 0
    correct_confidences: list[float] = []
    wrong_confidences: list[float] = []
    total = 0

    for path in paths[:1]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if max_raw_rows and total >= max_raw_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                if bool(record.get("parse_success", False)):
                    parse_success_count += 1
                confidence = _to_float(record.get("confidence"))
                if confidence is not None:
                    confidences.append(confidence)
                label_value = record.get("label")
                label = int(label_value) if str(label_value).strip() in {"0", "1"} else None
                if label is not None:
                    labels.append(label)
                decision = _normalise_decision(record.get("recommend"))
                decisions.append(decision)
                if confidence is not None and label is not None and decision in {"yes", "no"}:
                    predicted_label = 1 if decision == "yes" else 0
                    if predicted_label == label:
                        correct_confidences.append(confidence)
                    else:
                        wrong_confidences.append(confidence)

    confidence_counter = Counter(round(value, 6) for value in confidences)
    decision_counter = Counter(decisions)
    top_confidence_values = ";".join(
        f"{value:g}:{count}" for value, count in confidence_counter.most_common(5)
    )
    confidence_mean = mean(confidences) if confidences else ""
    confidence_std = pstdev(confidences) if len(confidences) > 1 else 0.0 if confidences else ""
    correct_mean = mean(correct_confidences) if correct_confidences else ""
    wrong_mean = mean(wrong_confidences) if wrong_confidences else ""
    confidence_gap = (
        float(correct_mean) - float(wrong_mean)
        if correct_mean != "" and wrong_mean != ""
        else ""
    )

    return {
        "prediction_file_status": "ready",
        "prediction_file": str(paths[0]),
        "raw_num_rows": total,
        "raw_parse_success_rate": _safe_ratio(parse_success_count, total),
        "raw_positive_label_ratio": _safe_ratio(sum(labels), len(labels)),
        "raw_decision_yes_ratio": _safe_ratio(decision_counter.get("yes", 0), len(decisions)),
        "raw_decision_no_ratio": _safe_ratio(decision_counter.get("no", 0), len(decisions)),
        "raw_confidence_mean": confidence_mean,
        "raw_confidence_std": confidence_std,
        "raw_confidence_min": min(confidences) if confidences else "",
        "raw_confidence_max": max(confidences) if confidences else "",
        "raw_confidence_unique_count": len(confidence_counter),
        "raw_confidence_eq_1_ratio": _safe_ratio(
            sum(1 for value in confidences if value == 1.0), len(confidences)
        ),
        "raw_confidence_ge_0_9_ratio": _safe_ratio(
            sum(1 for value in confidences if value >= 0.9), len(confidences)
        ),
        "raw_confidence_ge_0_8_ratio": _safe_ratio(
            sum(1 for value in confidences if value >= 0.8), len(confidences)
        ),
        "raw_confidence_le_0_1_ratio": _safe_ratio(
            sum(1 for value in confidences if value <= 0.1), len(confidences)
        ),
        "raw_top_confidence_values": top_confidence_values,
        "raw_correct_confidence_mean": correct_mean,
        "raw_wrong_confidence_mean": wrong_mean,
        "raw_confidence_gap_correct_minus_wrong": confidence_gap,
    }


def _collapse_label(row: dict[str, Any]) -> str:
    auroc = _to_float(row.get("auroc"))
    cal_auroc = _to_float(row.get("calibrated_auroc_after"))
    unique_count = _to_float(row.get("raw_confidence_unique_count"))
    eq_one = _to_float(row.get("raw_confidence_eq_1_ratio"))
    ge_09 = _to_float(row.get("raw_confidence_ge_0_9_ratio"))
    gap = _to_float(row.get("raw_confidence_gap_correct_minus_wrong"))

    flags: list[str] = []
    if auroc is not None and auroc <= 0.55:
        flags.append("weak_raw_auroc")
    if cal_auroc is not None and cal_auroc <= 0.60:
        flags.append("weak_calibrated_auroc")
    if unique_count is not None and unique_count <= 3:
        flags.append("low_score_diversity")
    if eq_one is not None and eq_one >= 0.80:
        flags.append("confidence_eq_1_collapse")
    if ge_09 is not None and ge_09 >= 0.80:
        flags.append("high_confidence_saturation")
    if gap is not None and abs(gap) <= 0.02:
        flags.append("low_correct_wrong_separation")
    return ";".join(flags) if flags else "not_obvious"


def _audit_spec(output_root: Path, spec: dict[str, str], max_raw_rows: int) -> dict[str, Any]:
    exp_dir = output_root / spec["exp_name"]
    tables_dir = exp_dir / "tables"
    diagnostic = _single_row_csv(tables_dir / "diagnostic_metrics.csv")
    calibration = _calibration_test_metrics(tables_dir / "calibration_comparison.csv")
    raw = _audit_raw_predictions(exp_dir, max_raw_rows)
    status = "ready" if diagnostic else "missing"

    row: dict[str, Any] = {
        "week_stage": "week7_9_light_audit",
        "signal_family": "light_old_pointwise_yesno_confidence",
        "route_family": spec["route_family"],
        "domain": spec["domain"],
        "scale_tag": spec["scale_tag"],
        "exp_name": spec["exp_name"],
        "status": status,
        "num_samples": diagnostic.get("num_samples", ""),
        "accuracy": diagnostic.get("accuracy", ""),
        "avg_confidence": diagnostic.get("avg_confidence", ""),
        "brier_score": diagnostic.get("brier_score", ""),
        "ece": diagnostic.get("ece", ""),
        "mce": diagnostic.get("mce", ""),
        "auroc": diagnostic.get("auroc", ""),
        "calibrated_ece_after": calibration.get("ece_after", ""),
        "calibrated_brier_after": calibration.get("brier_score_after", ""),
        "calibrated_auroc_after": calibration.get("auroc_after", ""),
    }
    row.update(raw)
    row["collapse_flags"] = _collapse_label(row) if status == "ready" else "missing"
    row["audit_takeaway"] = _takeaway(row)
    return row


def _takeaway(row: dict[str, Any]) -> str:
    if row.get("status") != "ready":
        return "missing_outputs"
    flags = str(row.get("collapse_flags", ""))
    if "confidence_eq_1_collapse" in flags:
        return "severe_verbalized_confidence_collapse"
    if "weak_raw_auroc" in flags and "high_confidence_saturation" in flags:
        return "weak_discrimination_with_high_confidence_saturation"
    if "weak_calibrated_auroc" in flags:
        return "calibration_improves_error_but_not_enough_discrimination"
    if "low_correct_wrong_separation" in flags:
        return "confidence_barely_separates_correct_from_wrong"
    return "usable_but_should_be_compared_against_shadow_signal"


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    parsed = _to_float(value)
    if parsed is None:
        return str(value or "")
    return f"{parsed:.4f}"


def _write_md(rows: list[dict[str, Any]], output_path: Path) -> Path:
    md_path = output_path.with_suffix(".md")
    lines = [
        "# Week7.9 Light Pointwise Signal Audit",
        "",
        "This audit checks the old/light pointwise signal: yes/no recommendation plus verbalized self-confidence. It is intentionally diagnostic, not a new method result.",
        "",
        "| route | domain | scale | status | acc | raw AUROC | cal AUROC | ECE | cal ECE | avg conf | conf>=0.9 | unique conf | flags |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("route_family", "")),
                    str(row.get("domain", "")),
                    str(row.get("scale_tag", "")),
                    str(row.get("status", "")),
                    _fmt(row.get("accuracy")),
                    _fmt(row.get("auroc")),
                    _fmt(row.get("calibrated_auroc_after")),
                    _fmt(row.get("ece")),
                    _fmt(row.get("calibrated_ece_after")),
                    _fmt(row.get("avg_confidence")),
                    _fmt(row.get("raw_confidence_ge_0_9_ratio")),
                    _fmt(row.get("raw_confidence_unique_count")),
                    str(row.get("collapse_flags", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Audit reading:",
            "",
            "- `weak_raw_auroc` means the old confidence score has weak sample-level discrimination before calibration.",
            "- `weak_calibrated_auroc` means calibration improves aggregate reliability but still does not make the score strongly rank correct vs wrong cases.",
            "- `high_confidence_saturation` and `confidence_eq_1_collapse` indicate verbalized confidence is concentrated near the top end.",
            "- These flags justify moving from light yes/no confidence toward shadow relevance/probability-style signals.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_path = Path(args.output_path)
    rows = [_audit_spec(output_root, spec, args.max_raw_rows) for spec in DEFAULT_SPECS]
    _write_csv(rows, output_path)
    md_path = _write_md(rows, output_path)
    print(f"Saved light pointwise audit CSV to: {output_path}")
    print(f"Saved light pointwise audit markdown to: {md_path}")
    for row in rows:
        print(
            f"{row['route_family']} {row['domain']} {row['scale_tag']}: "
            f"{row['status']} auroc={_fmt(row.get('auroc'))} "
            f"cal_auroc={_fmt(row.get('calibrated_auroc_after'))} "
            f"flags={row.get('collapse_flags')}"
        )


if __name__ == "__main__":
    main()

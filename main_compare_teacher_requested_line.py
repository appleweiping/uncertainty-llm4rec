from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


DEFAULT_POINTWISE_SPECS: list[dict[str, str]] = [
    {
        "domain": "beauty",
        "replay_exp_name": "beauty_qwen3_local_replay_v2_pointwise_full",
        "historical_exp_name": "beauty_deepseek_pointwise_full",
    },
    {
        "domain": "books",
        "replay_exp_name": "books_qwen3_local_replay_v2_pointwise_full",
        "historical_exp_name": "books_deepseek_pointwise_full3000",
    },
    {
        "domain": "electronics",
        "replay_exp_name": "electronics_qwen3_local_replay_v2_pointwise_full",
        "historical_exp_name": "electronics_deepseek_pointwise_full3000",
    },
    {
        "domain": "movies",
        "replay_exp_name": "movies_qwen3_local_replay_v2_pointwise_full",
        "historical_exp_name": "movies_deepseek_pointwise_full3000",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Week7.8 teacher-requested local-8B LoRA replay summaries."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="pointwise",
        choices=["pointwise"],
        help="Current supported summary mode.",
    )
    parser.add_argument("--output_root", type=str, default="outputs", help="Experiment output root.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/summary/week7_8_replay_v2_week1_week2_pointwise_summary.csv",
        help="Summary CSV output path.",
    )
    return parser.parse_args()


def _single_row_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    return rows[0]


def _calibration_after_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    metrics: dict[str, Any] = {}
    for row in rows:
        if str(row.get("split", "")).strip().lower() != "test":
            continue
        metric_name = str(row.get("metric", "")).strip().lower().replace("@", "_at_").replace("-", "_")
        if not metric_name:
            continue
        metrics[f"{metric_name}_before"] = row.get("before")
        metrics[f"{metric_name}_after"] = row.get("after")
    return metrics


def _load_pointwise_bundle(output_root: Path, exp_name: str) -> dict[str, Any]:
    exp_dir = output_root / exp_name
    tables_dir = exp_dir / "tables"
    diagnostic = _single_row_csv(tables_dir / "diagnostic_metrics.csv")
    confidence = _single_row_csv(tables_dir / "confidence_correctness_summary.csv")
    calibration = _calibration_after_metrics(tables_dir / "calibration_comparison.csv")
    status = "ready" if diagnostic else "missing"
    return {
        "status": status,
        "diagnostic": diagnostic,
        "confidence": confidence,
        "calibration": calibration,
    }


def _row_from_spec(output_root: Path, spec: dict[str, str]) -> dict[str, Any]:
    replay = _load_pointwise_bundle(output_root, spec["replay_exp_name"])
    historical = _load_pointwise_bundle(output_root, spec["historical_exp_name"])

    row: dict[str, Any] = {
        "week_stage": "week7_8_replay",
        "route_role": "teacher_requested_local_mainline",
        "domain": spec["domain"],
        "replay_exp_name": spec["replay_exp_name"],
        "historical_exp_name": spec["historical_exp_name"],
        "replay_status": replay["status"],
        "historical_status": historical["status"],
    }

    replay_diag = replay["diagnostic"]
    replay_conf = replay["confidence"]
    replay_cal = replay["calibration"]
    hist_diag = historical["diagnostic"]

    row.update(
        {
            "replay_num_samples": replay_diag.get("num_samples"),
            "replay_accuracy": replay_diag.get("accuracy"),
            "replay_avg_confidence": replay_diag.get("avg_confidence"),
            "replay_brier_score": replay_diag.get("brier_score"),
            "replay_ece": replay_diag.get("ece"),
            "replay_auroc": replay_diag.get("auroc"),
            "replay_high_conf_ratio": replay_conf.get("high_conf_ratio"),
            "replay_high_conf_accuracy": replay_conf.get("high_conf_accuracy"),
            "replay_calibration_ece_before": replay_cal.get("ece_before"),
            "replay_calibration_ece_after": replay_cal.get("ece_after"),
            "replay_calibration_brier_score_before": replay_cal.get("brier_score_before"),
            "replay_calibration_brier_score_after": replay_cal.get("brier_score_after"),
            "historical_num_samples": hist_diag.get("num_samples"),
            "historical_accuracy": hist_diag.get("accuracy"),
            "historical_avg_confidence": hist_diag.get("avg_confidence"),
            "historical_brier_score": hist_diag.get("brier_score"),
            "historical_ece": hist_diag.get("ece"),
            "historical_auroc": hist_diag.get("auroc"),
        }
    )
    replay_ece = _to_float(row.get("replay_ece"))
    historical_ece = _to_float(row.get("historical_ece"))
    replay_acc = _to_float(row.get("replay_accuracy"))
    historical_acc = _to_float(row.get("historical_accuracy"))
    row["delta_ece_vs_historical"] = (
        replay_ece - historical_ece if replay_ece is not None and historical_ece is not None else ""
    )
    row["delta_accuracy_vs_historical"] = (
        replay_acc - historical_acc if replay_acc is not None and historical_acc is not None else ""
    )
    return row


def _write_md(summary_rows: list[dict[str, Any]], output_path: Path) -> Path:
    md_path = output_path.with_suffix(".md")
    lines = [
        "# Week7.8 Replay Pointwise Summary",
        "",
        "This table tracks the teacher-requested local-v2 replay pointwise/calibration line against the preserved historical teacher-observation route.",
        "",
        "| domain | replay_status | historical_status | replay_accuracy | replay_ece | historical_accuracy | historical_ece |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("domain", "")),
                    str(row.get("replay_status", "")),
                    str(row.get("historical_status", "")),
                    _fmt(row.get("replay_accuracy")),
                    _fmt(row.get("replay_ece")),
                    _fmt(row.get("historical_accuracy")),
                    _fmt(row.get("historical_ece")),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (float, int)):
        return f"{float(value):.6f}"
    return str(value)


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_path = Path(args.output_path)
    rows = [_row_from_spec(output_root, spec) for spec in DEFAULT_POINTWISE_SPECS]
    _write_csv(rows, output_path)
    md_path = _write_md(rows, output_path)
    print(f"Saved Week7.8 teacher-requested summary to: {output_path}")
    print(f"Saved markdown handoff to: {md_path}")
    for row in rows:
        print(
            f"{row['domain']}: replay={row['replay_status']} "
            f"historical={row['historical_status']} "
            f"replay_acc={_fmt(row.get('replay_accuracy'))} "
            f"replay_ece={_fmt(row.get('replay_ece'))}"
        )


if __name__ == "__main__":
    main()

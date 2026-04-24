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

DEFAULT_RERANK_SPECS: list[dict[str, str]] = [
    {
        "domain": "beauty",
        "replay_rank_exp_name": "beauty_qwen3_local_replay_v2_rank_full973",
        "replay_rerank_exp_name": "beauty_qwen3_local_replay_v2_structured_risk_full973",
        "reference_exp_name": "beauty_deepseek_rank_full973_structured_risk",
    },
    {
        "domain": "books",
        "replay_rank_exp_name": "books_qwen3_local_replay_v2_rank_full",
        "replay_rerank_exp_name": "books_qwen3_local_replay_v2_structured_risk_full",
        "reference_exp_name": "books_deepseek_rank_full500_structured_risk",
    },
    {
        "domain": "electronics",
        "replay_rank_exp_name": "electronics_qwen3_local_replay_v2_rank_full",
        "replay_rerank_exp_name": "electronics_qwen3_local_replay_v2_structured_risk_full",
        "reference_exp_name": "electronics_deepseek_rank_full500_structured_risk",
    },
    {
        "domain": "movies",
        "replay_rank_exp_name": "movies_qwen3_local_replay_v2_rank_full",
        "replay_rerank_exp_name": "movies_qwen3_local_replay_v2_structured_risk_full",
        "reference_exp_name": "movies_deepseek_rank_full500_structured_risk",
    },
]

DEFAULT_ROBUSTNESS_SPECS: list[dict[str, str]] = [
    {
        "domain": "beauty",
        "clean_exp_name": "beauty_qwen3_local_replay_v2_structured_risk_full973",
        "noisy_exp_name": "beauty_qwen3_local_replay_v2_structured_risk_full973_noisy_nl10",
    },
    {
        "domain": "books",
        "clean_exp_name": "books_qwen3_local_replay_v2_structured_risk_full",
        "noisy_exp_name": "books_qwen3_local_replay_v2_structured_risk_full_noisy_nl10",
    },
    {
        "domain": "electronics",
        "clean_exp_name": "electronics_qwen3_local_replay_v2_structured_risk_full",
        "noisy_exp_name": "electronics_qwen3_local_replay_v2_structured_risk_full_noisy_nl10",
    },
    {
        "domain": "movies",
        "clean_exp_name": "movies_qwen3_local_replay_v2_structured_risk_full",
        "noisy_exp_name": "movies_qwen3_local_replay_v2_structured_risk_full_noisy_nl10",
    },
]

LIGHT_EXECUTABLE_POINTWISE_SPECS: list[dict[str, str]] = [
    {
        "domain": "beauty",
        "replay_exp_name": "beauty_qwen3_local_replay_v2_pointwise_full",
        "historical_exp_name": "beauty_deepseek_pointwise_full",
    },
    {
        "domain": "books",
        "replay_exp_name": "books_qwen3_local_pointwise_small3000",
        "historical_exp_name": "books_deepseek_pointwise_full3000",
    },
    {
        "domain": "electronics",
        "replay_exp_name": "electronics_qwen3_local_pointwise_small3000",
        "historical_exp_name": "electronics_deepseek_pointwise_full3000",
    },
    {
        "domain": "movies",
        "replay_exp_name": "movies_qwen3_local_pointwise_small3000",
        "historical_exp_name": "movies_deepseek_pointwise_full3000",
    },
]

LIGHT_EXECUTABLE_RERANK_SPECS: list[dict[str, str]] = [
    {
        "domain": "beauty",
        "replay_rank_exp_name": "beauty_qwen3_local_replay_v2_rank_full973",
        "replay_rerank_exp_name": "beauty_qwen3_local_replay_v2_structured_risk_full973",
        "reference_exp_name": "beauty_deepseek_rank_full973_structured_risk",
    },
    {
        "domain": "books",
        "replay_rank_exp_name": "books_qwen3_local_rank_small500",
        "replay_rerank_exp_name": "books_qwen3_local_structured_risk_small500",
        "reference_exp_name": "books_deepseek_rank_full500_structured_risk",
    },
    {
        "domain": "electronics",
        "replay_rank_exp_name": "electronics_qwen3_local_rank_small500",
        "replay_rerank_exp_name": "electronics_qwen3_local_structured_risk_small500",
        "reference_exp_name": "electronics_deepseek_rank_full500_structured_risk",
    },
    {
        "domain": "movies",
        "replay_rank_exp_name": "movies_qwen3_local_rank_small500",
        "replay_rerank_exp_name": "movies_qwen3_local_structured_risk_small500",
        "reference_exp_name": "movies_deepseek_rank_full500_structured_risk",
    },
]

LIGHT_EXECUTABLE_ROBUSTNESS_SPECS: list[dict[str, str]] = [
    {
        "domain": "beauty",
        "clean_exp_name": "beauty_qwen3_local_replay_v2_structured_risk_full973",
        "noisy_exp_name": "beauty_qwen3_local_replay_v2_structured_risk_full973_noisy_nl10",
    },
    {
        "domain": "books",
        "clean_exp_name": "books_qwen3_local_structured_risk_small500",
        "noisy_exp_name": "books_qwen3_local_structured_risk_small500_noisy_nl10",
    },
    {
        "domain": "electronics",
        "clean_exp_name": "electronics_qwen3_local_structured_risk_small500",
        "noisy_exp_name": "electronics_qwen3_local_structured_risk_small500_noisy_nl10",
    },
    {
        "domain": "movies",
        "clean_exp_name": "movies_qwen3_local_structured_risk_small500",
        "noisy_exp_name": "movies_qwen3_local_structured_risk_small500_noisy_nl10",
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
        choices=[
            "pointwise",
            "rerank",
            "robustness",
            "final",
            "light_pointwise",
            "light_rerank",
            "light_robustness",
            "light_final",
        ],
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


def _multi_row_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def _metric_from_row(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key, "")
        if value not in {"", None}:
            return value
    return ""


def _find_method_row(rows: list[dict[str, Any]], method_name: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("method", "")).strip().lower() == method_name.lower():
            return row
    return {}


def _load_robustness_summary(output_root: Path, clean_exp_name: str, noisy_exp_name: str) -> dict[str, Any]:
    compare_name = f"{clean_exp_name}_vs_{noisy_exp_name}"
    path = output_root / "robustness" / compare_name / "tables" / "robustness_summary.csv"
    row = _single_row_csv(path)
    return {
        "status": "ready" if row else "missing",
        "compare_name": compare_name,
        "row": row,
    }


def _load_summary_rows(path: Path) -> list[dict[str, Any]]:
    return _multi_row_csv(path)


def _rows_by_domain(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        domain = str(row.get("domain", "")).strip().lower()
        if domain:
            mapping[domain] = row
    return mapping


def _load_direct_bundle(output_root: Path, exp_name: str) -> dict[str, Any]:
    rows = _multi_row_csv(output_root / exp_name / "tables" / "ranking_metrics.csv")
    if not rows:
        return {"status": "missing", "row": {}}
    direct_row = rows[0]
    return {"status": "ready", "row": direct_row}


def _load_rerank_bundle(output_root: Path, exp_name: str) -> dict[str, Any]:
    rows = _multi_row_csv(output_root / exp_name / "tables" / "rerank_results.csv")
    if not rows:
        return {"status": "missing", "direct_row": {}, "rerank_row": {}}
    direct_row = _find_method_row(rows, "direct_candidate_ranking")
    rerank_row = _find_method_row(rows, "uncertainty_aware_rank_rerank_nonlinear_structured_risk_rerank")
    if not rerank_row and rows:
        rerank_row = rows[-1]
    return {"status": "ready", "direct_row": direct_row, "rerank_row": rerank_row}


def _rerank_row_from_spec(output_root: Path, spec: dict[str, str]) -> dict[str, Any]:
    replay_direct = _load_direct_bundle(output_root, spec["replay_rank_exp_name"])
    replay_rerank = _load_rerank_bundle(output_root, spec["replay_rerank_exp_name"])
    reference = _load_rerank_bundle(output_root, spec["reference_exp_name"])

    replay_direct_status = replay_direct["status"]
    replay_direct_row = replay_direct["row"]
    if replay_direct_status == "missing" and replay_rerank["direct_row"]:
        replay_direct_status = "ready_from_rerank_baseline"
        replay_direct_row = replay_rerank["direct_row"]
    replay_rerank_row = replay_rerank["rerank_row"]
    reference_rerank_row = reference["rerank_row"]

    row: dict[str, Any] = {
        "week_stage": "week7_8_replay",
        "route_role": "teacher_requested_local_mainline",
        "domain": spec["domain"],
        "replay_rank_exp_name": spec["replay_rank_exp_name"],
        "replay_rerank_exp_name": spec["replay_rerank_exp_name"],
        "reference_exp_name": spec["reference_exp_name"],
        "replay_direct_status": replay_direct_status,
        "replay_rerank_status": replay_rerank["status"],
        "reference_status": reference["status"],
        "replay_direct_hr_at_10": _metric_from_row(replay_direct_row, "HR@10"),
        "replay_direct_ndcg_at_10": _metric_from_row(replay_direct_row, "NDCG@10"),
        "replay_direct_mrr": _metric_from_row(replay_direct_row, "MRR"),
        "replay_direct_coverage_at_10": _metric_from_row(replay_direct_row, "coverage@10", "coverage"),
        "replay_direct_head_exposure_ratio_at_10": _metric_from_row(replay_direct_row, "head_exposure_ratio@10", "head_exposure_ratio"),
        "replay_direct_longtail_coverage_at_10": _metric_from_row(replay_direct_row, "longtail_coverage@10", "longtail_coverage"),
        "replay_rerank_hr_at_10": _metric_from_row(replay_rerank_row, "HR@10"),
        "replay_rerank_ndcg_at_10": _metric_from_row(replay_rerank_row, "NDCG@10"),
        "replay_rerank_mrr": _metric_from_row(replay_rerank_row, "MRR"),
        "replay_rerank_coverage_at_10": _metric_from_row(replay_rerank_row, "coverage@10", "coverage"),
        "replay_rerank_head_exposure_ratio_at_10": _metric_from_row(replay_rerank_row, "head_exposure_ratio@10", "head_exposure_ratio"),
        "replay_rerank_longtail_coverage_at_10": _metric_from_row(replay_rerank_row, "longtail_coverage@10", "longtail_coverage"),
        "reference_hr_at_10": _metric_from_row(reference_rerank_row, "HR@10"),
        "reference_ndcg_at_10": _metric_from_row(reference_rerank_row, "NDCG@10"),
        "reference_mrr": _metric_from_row(reference_rerank_row, "MRR"),
        "reference_coverage_at_10": _metric_from_row(reference_rerank_row, "coverage@10", "coverage"),
        "reference_head_exposure_ratio_at_10": _metric_from_row(reference_rerank_row, "head_exposure_ratio@10", "head_exposure_ratio"),
        "reference_longtail_coverage_at_10": _metric_from_row(reference_rerank_row, "longtail_coverage@10", "longtail_coverage"),
    }
    direct_ndcg = _to_float(row.get("replay_direct_ndcg_at_10"))
    rerank_ndcg = _to_float(row.get("replay_rerank_ndcg_at_10"))
    ref_ndcg = _to_float(row.get("reference_ndcg_at_10"))
    row["delta_rerank_vs_direct_ndcg_at_10"] = (
        rerank_ndcg - direct_ndcg if rerank_ndcg is not None and direct_ndcg is not None else ""
    )
    row["delta_rerank_vs_reference_ndcg_at_10"] = (
        rerank_ndcg - ref_ndcg if rerank_ndcg is not None and ref_ndcg is not None else ""
    )
    return row


def _robustness_row_from_spec(output_root: Path, spec: dict[str, str]) -> dict[str, Any]:
    clean_bundle = _load_rerank_bundle(output_root, spec["clean_exp_name"])
    robustness_bundle = _load_robustness_summary(output_root, spec["clean_exp_name"], spec["noisy_exp_name"])
    clean_row = clean_bundle["rerank_row"]
    robustness_row = robustness_bundle["row"]

    row: dict[str, Any] = {
        "week_stage": "week7_8_replay",
        "route_role": "teacher_requested_local_mainline",
        "domain": spec["domain"],
        "clean_exp_name": spec["clean_exp_name"],
        "noisy_exp_name": spec["noisy_exp_name"],
        "robustness_compare_name": robustness_bundle["compare_name"],
        "clean_status": clean_bundle["status"],
        "robustness_status": robustness_bundle["status"],
        "clean_hr_at_10": _metric_from_row(clean_row, "HR@10"),
        "clean_ndcg_at_10": _metric_from_row(clean_row, "NDCG@10"),
        "clean_mrr": _metric_from_row(clean_row, "MRR"),
        "clean_coverage_at_10": _metric_from_row(clean_row, "coverage@10", "coverage"),
        "clean_head_exposure_ratio_at_10": _metric_from_row(clean_row, "head_exposure_ratio@10", "head_exposure_ratio"),
        "clean_longtail_coverage_at_10": _metric_from_row(clean_row, "longtail_coverage@10", "longtail_coverage"),
        "clean_parse_success_rate": _metric_from_row(clean_row, "parse_success_rate"),
        "clean_out_of_candidate_rate": _metric_from_row(clean_row, "out_of_candidate_rate"),
        "noise_level": _metric_from_row(robustness_row, "noise_level"),
        "hr_at_10_drop": _metric_from_row(robustness_row, "rerank_HR@10_drop"),
        "ndcg_at_10_drop": _metric_from_row(robustness_row, "rerank_NDCG@10_drop"),
        "mrr_drop": _metric_from_row(robustness_row, "rerank_MRR@10_drop"),
        "head_exposure_ratio_at_10_drop": _metric_from_row(robustness_row, "rerank_head_exposure_ratio@10_drop"),
        "long_tail_coverage_at_10_drop": _metric_from_row(robustness_row, "rerank_long_tail_coverage@10_drop"),
        "lambda_penalty_drop": _metric_from_row(robustness_row, "rerank_lambda_penalty_drop"),
    }
    return row


def _final_rows(
    output_root: Path,
    *,
    pointwise_summary_filename: str = "week7_8_replay_v2_week1_week2_pointwise_summary.csv",
    rerank_summary_filename: str = "week7_8_replay_v2_week3_rerank_compare.csv",
    robustness_summary_filename: str = "week7_8_replay_v2_week4_robustness_summary.csv",
) -> list[dict[str, Any]]:
    summary_dir = output_root / "summary"
    pointwise_rows = _rows_by_domain(
        _load_summary_rows(summary_dir / pointwise_summary_filename)
    )
    rerank_rows = _rows_by_domain(
        _load_summary_rows(summary_dir / rerank_summary_filename)
    )
    robustness_rows = _rows_by_domain(
        _load_summary_rows(summary_dir / robustness_summary_filename)
    )

    final_rows: list[dict[str, Any]] = []
    for domain in ["beauty", "books", "electronics", "movies"]:
        pointwise = pointwise_rows.get(domain, {})
        rerank = rerank_rows.get(domain, {})
        robustness = robustness_rows.get(domain, {})

        replay_pointwise_status = str(pointwise.get("replay_status", "missing") or "missing")
        replay_direct_status = str(rerank.get("replay_direct_status", "missing") or "missing")
        replay_rerank_status = str(rerank.get("replay_rerank_status", "missing") or "missing")
        replay_robustness_status = str(robustness.get("robustness_status", "missing") or "missing")
        historical_status = str(pointwise.get("historical_status", "missing") or "missing")
        reference_status = str(rerank.get("reference_status", "missing") or "missing")

        all_replay_ready = all(
            status == "ready"
            for status in [
                replay_pointwise_status,
                replay_direct_status,
                replay_rerank_status,
                replay_robustness_status,
            ]
        )

        final_rows.append(
            {
                "week_stage": "week7_8_replay",
                "route_role": "teacher_requested_local_mainline",
                "domain": domain,
                "historical_teacher_status": historical_status,
                "replay_pointwise_status": replay_pointwise_status,
                "replay_direct_status": replay_direct_status,
                "replay_rerank_status": replay_rerank_status,
                "replay_robustness_status": replay_robustness_status,
                "structured_risk_reference_status": reference_status,
                "all_replay_layers_ready": "ready" if all_replay_ready else "missing",
                "historical_accuracy": pointwise.get("historical_accuracy", ""),
                "historical_ece": pointwise.get("historical_ece", ""),
                "historical_brier_score": pointwise.get("historical_brier_score", ""),
                "reference_ndcg_at_10": rerank.get("reference_ndcg_at_10", ""),
                "reference_mrr": rerank.get("reference_mrr", ""),
                "reference_coverage_at_10": rerank.get("reference_coverage_at_10", ""),
                "reference_head_exposure_ratio_at_10": rerank.get("reference_head_exposure_ratio_at_10", ""),
                "reference_longtail_coverage_at_10": rerank.get("reference_longtail_coverage_at_10", ""),
                "replay_pointwise_exp_name": pointwise.get("replay_exp_name", ""),
                "replay_rank_exp_name": rerank.get("replay_rank_exp_name", ""),
                "replay_rerank_exp_name": rerank.get("replay_rerank_exp_name", ""),
                "replay_robustness_compare_name": robustness.get("robustness_compare_name", ""),
                "replay_clean_exp_name": robustness.get("clean_exp_name", ""),
                "replay_noisy_exp_name": robustness.get("noisy_exp_name", ""),
            }
        )
    return final_rows


def _write_pointwise_md(summary_rows: list[dict[str, Any]], output_path: Path) -> Path:
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


def _write_rerank_md(summary_rows: list[dict[str, Any]], output_path: Path) -> Path:
    md_path = output_path.with_suffix(".md")
    lines = [
        "# Week7.8 Replay Day3 Ranking Rerank Compare",
        "",
        "This table tracks the teacher-requested local-v2 ranking and rerank replay line against the preserved strongest hand-crafted structured-risk reference.",
        "",
        "| domain | replay_direct_status | replay_rerank_status | reference_status | replay_direct_ndcg@10 | replay_rerank_ndcg@10 | reference_ndcg@10 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("domain", "")),
                    str(row.get("replay_direct_status", "")),
                    str(row.get("replay_rerank_status", "")),
                    str(row.get("reference_status", "")),
                    _fmt(row.get("replay_direct_ndcg_at_10")),
                    _fmt(row.get("replay_rerank_ndcg_at_10")),
                    _fmt(row.get("reference_ndcg_at_10")),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def _write_robustness_md(summary_rows: list[dict[str, Any]], output_path: Path) -> Path:
    md_path = output_path.with_suffix(".md")
    lines = [
        "# Week7.8 Replay Day4 Robustness Summary",
        "",
        "This table tracks the teacher-requested local-v2 clean robustness baseline and the status of the minimal clean/noisy replay compare.",
        "",
        "| domain | clean_status | robustness_status | clean_ndcg@10 | clean_coverage@10 | clean_head_exposure_ratio@10 | ndcg@10_drop |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("domain", "")),
                    str(row.get("clean_status", "")),
                    str(row.get("robustness_status", "")),
                    _fmt(row.get("clean_ndcg_at_10")),
                    _fmt(row.get("clean_coverage_at_10")),
                    _fmt(row.get("clean_head_exposure_ratio_at_10")),
                    _fmt(row.get("ndcg_at_10_drop")),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def _write_final_md(summary_rows: list[dict[str, Any]], output_path: Path) -> Path:
    md_path = output_path.with_suffix(".md")
    lines = [
        "# Teacher Requested Local 8B LoRA Mainline Final",
        "",
        "This table is the Week7.8 final handoff for the teacher-requested local-v2 replay mainline. It preserves the historical official-API teacher evidence, the structured-risk strongest hand-crafted reference, and the current readiness state of the replay pointwise, decision, and robustness layers.",
        "",
        "| domain | historical_teacher_status | replay_pointwise_status | replay_direct_status | replay_rerank_status | replay_robustness_status | structured_risk_reference_status | all_replay_layers_ready | reference_ndcg@10 | historical_ece |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("domain", "")),
                    str(row.get("historical_teacher_status", "")),
                    str(row.get("replay_pointwise_status", "")),
                    str(row.get("replay_direct_status", "")),
                    str(row.get("replay_rerank_status", "")),
                    str(row.get("replay_robustness_status", "")),
                    str(row.get("structured_risk_reference_status", "")),
                    str(row.get("all_replay_layers_ready", "")),
                    _fmt(row.get("reference_ndcg_at_10")),
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
    default_pointwise_output = Path("outputs/summary/week7_8_replay_v2_week1_week2_pointwise_summary.csv")
    if args.mode == "pointwise":
        rows = [_row_from_spec(output_root, spec) for spec in DEFAULT_POINTWISE_SPECS]
    elif args.mode == "light_pointwise":
        if output_path.as_posix() == default_pointwise_output.as_posix():
            output_path = Path("outputs/summary/week7_8_light_week1_week2_pointwise_summary.csv")
        rows = [_row_from_spec(output_root, spec) for spec in LIGHT_EXECUTABLE_POINTWISE_SPECS]
    elif args.mode == "rerank":
        if output_path.as_posix() == default_pointwise_output.as_posix():
            output_path = Path("outputs/summary/week7_8_replay_v2_week3_rerank_compare.csv")
        rows = [_rerank_row_from_spec(output_root, spec) for spec in DEFAULT_RERANK_SPECS]
    elif args.mode == "light_rerank":
        if output_path.as_posix() == default_pointwise_output.as_posix():
            output_path = Path("outputs/summary/week7_8_light_week3_rerank_compare.csv")
        rows = [_rerank_row_from_spec(output_root, spec) for spec in LIGHT_EXECUTABLE_RERANK_SPECS]
    elif args.mode == "robustness":
        if output_path.as_posix() == default_pointwise_output.as_posix():
            output_path = Path("outputs/summary/week7_8_replay_v2_week4_robustness_summary.csv")
        rows = [_robustness_row_from_spec(output_root, spec) for spec in DEFAULT_ROBUSTNESS_SPECS]
    elif args.mode == "light_robustness":
        if output_path.as_posix() == default_pointwise_output.as_posix():
            output_path = Path("outputs/summary/week7_8_light_week4_robustness_summary.csv")
        rows = [_robustness_row_from_spec(output_root, spec) for spec in LIGHT_EXECUTABLE_ROBUSTNESS_SPECS]
    elif args.mode == "final":
        if output_path.as_posix() == default_pointwise_output.as_posix():
            output_path = Path("outputs/summary/teacher_requested_local8b_lora_mainline_final.csv")
        rows = _final_rows(output_root)
    else:
        if output_path.as_posix() == default_pointwise_output.as_posix():
            output_path = Path("outputs/summary/week7_8_light_teacher_requested_mainline_final.csv")
        rows = _final_rows(
            output_root,
            pointwise_summary_filename="week7_8_light_week1_week2_pointwise_summary.csv",
            rerank_summary_filename="week7_8_light_week3_rerank_compare.csv",
            robustness_summary_filename="week7_8_light_week4_robustness_summary.csv",
        )
    _write_csv(rows, output_path)
    if args.mode in {"pointwise", "light_pointwise"}:
        md_path = _write_pointwise_md(rows, output_path)
    elif args.mode in {"rerank", "light_rerank"}:
        md_path = _write_rerank_md(rows, output_path)
    elif args.mode in {"robustness", "light_robustness"}:
        md_path = _write_robustness_md(rows, output_path)
    else:
        md_path = _write_final_md(rows, output_path)
    print(f"Saved Week7.8 teacher-requested summary to: {output_path}")
    print(f"Saved markdown handoff to: {md_path}")
    for row in rows:
        if args.mode in {"pointwise", "light_pointwise"}:
            print(
                f"{row['domain']}: replay={row['replay_status']} "
                f"historical={row['historical_status']} "
                f"replay_acc={_fmt(row.get('replay_accuracy'))} "
                f"replay_ece={_fmt(row.get('replay_ece'))}"
            )
        elif args.mode in {"rerank", "light_rerank"}:
            print(
                f"{row['domain']}: direct={row['replay_direct_status']} "
                f"rerank={row['replay_rerank_status']} "
                f"reference={row['reference_status']} "
                f"rerank_ndcg={_fmt(row.get('replay_rerank_ndcg_at_10'))}"
            )
        elif args.mode in {"robustness", "light_robustness"}:
            print(
                f"{row['domain']}: clean={row['clean_status']} "
                f"robustness={row['robustness_status']} "
                f"clean_ndcg={_fmt(row.get('clean_ndcg_at_10'))} "
                f"ndcg_drop={_fmt(row.get('ndcg_at_10_drop'))}"
            )
        else:
            print(
                f"{row['domain']}: pointwise={row['replay_pointwise_status']} "
                f"direct={row['replay_direct_status']} "
                f"rerank={row['replay_rerank_status']} "
                f"robustness={row['replay_robustness_status']} "
                f"final={row['all_replay_layers_ready']}"
            )


if __name__ == "__main__":
    main()

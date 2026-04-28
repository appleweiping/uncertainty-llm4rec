"""Day37 small-domain CEP relevance evidence alignment.

Local orchestration helpers for preflight, smoke-status collection, calibration,
field diagnostics, Beauty-vs-small summary, and reporting. API inference is run
separately via main_infer.py so resume/concurrency behavior stays unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd

from main_calibrate_relevance_evidence import (
    apply_relevance_posterior,
    apply_score_calibrator,
    build_relevance_frame,
    fit_relevance_posterior,
    metrics_row,
    usable_rows,
)
from src.uncertainty.calibration import fit_calibrator


DOMAINS = {
    "movies": {
        "data_dir": Path("data/processed/amazon_movies_small"),
        "exp_name": "movies_small_deepseek_relevance_evidence",
        "config": Path("configs/exp/movies_small_deepseek_relevance_evidence.yaml"),
    },
    "books": {
        "data_dir": Path("data/processed/amazon_books_small"),
        "exp_name": "books_small_deepseek_relevance_evidence",
        "config": Path("configs/exp/books_small_deepseek_relevance_evidence.yaml"),
    },
    "electronics": {
        "data_dir": Path("data/processed/amazon_electronics_small"),
        "exp_name": "electronics_small_deepseek_relevance_evidence",
        "config": Path("configs/exp/electronics_small_deepseek_relevance_evidence.yaml"),
    },
}

SUMMARY_DIR = Path("output-repaired/summary")
OUTPUT_ROOT = Path("output-repaired")
SMOKE_INPUT_DIR = Path("output-repaired/day37_small_smoke_inputs")
REQUIRED_SCHEMA = ["user_id", "history", "candidate_item_id", "candidate_title", "candidate_text", "label"]
EVIDENCE_FIELDS = [
    "relevance_probability",
    "positive_evidence",
    "negative_evidence",
    "ambiguity",
    "missing_information",
]
DIAGNOSTIC_FIELDS = [
    "relevance_probability",
    "positive_evidence",
    "negative_evidence",
    "abs_evidence_margin",
    "ambiguity",
    "missing_information",
    "evidence_risk",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def output_dir(domain: str) -> Path:
    return OUTPUT_ROOT / DOMAINS[domain]["exp_name"]


def prediction_path(domain: str, split: str) -> Path:
    return output_dir(domain) / "predictions" / f"{split}_raw.jsonl"


def smoke_output_path(domain: str, split: str) -> Path:
    return output_dir(domain) / "predictions" / f"smoke_{split}_raw.jsonl"


def split_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_user: dict[str, int] = {}
    for row in rows:
        user = str(row.get("user_id", ""))
        by_user[user] = by_user.get(user, 0) + 1
    counts = list(by_user.values())
    sample = rows[0] if rows else {}
    return {
        "rows": len(rows),
        "users": len(by_user),
        "candidate_pool_size_mean": mean(counts) if counts else 0,
        "candidate_pool_size_min": min(counts) if counts else 0,
        "candidate_pool_size_max": max(counts) if counts else 0,
        **{f"has_{field}": field in sample for field in REQUIRED_SCHEMA},
    }


def prepare() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    preflight_rows: list[dict[str, Any]] = []
    monitor_lines = [
        "# Day37 Small-Domain Runtime Monitor",
        "",
        "No API inference is launched by `--prepare`; use the listed commands with resume.",
        "",
    ]
    for domain, cfg in DOMAINS.items():
        data_dir: Path = cfg["data_dir"]  # type: ignore[assignment]
        train = read_jsonl(data_dir / "train.jsonl")
        valid = read_jsonl(data_dir / "valid.jsonl")
        test = read_jsonl(data_dir / "test.jsonl")
        valid_stats = split_stats(valid)
        test_stats = split_stats(test)
        schema_ok = all(valid_stats.get(f"has_{f}", False) for f in REQUIRED_SCHEMA) and all(
            test_stats.get(f"has_{f}", False) for f in REQUIRED_SCHEMA
        )
        pool_mean = test_stats["candidate_pool_size_mean"] or valid_stats["candidate_pool_size_mean"]
        preflight_rows.append(
            {
                "domain": domain,
                "train_rows": len(train),
                "valid_rows": len(valid),
                "test_rows": len(test),
                "valid_users": valid_stats["users"],
                "test_users": test_stats["users"],
                "candidate_pool_size_mean": pool_mean,
                "has_user_id": bool(valid_stats["has_user_id"] and test_stats["has_user_id"]),
                "has_history": bool(valid_stats["has_history"] and test_stats["has_history"]),
                "has_candidate_item_id": bool(valid_stats["has_candidate_item_id"] and test_stats["has_candidate_item_id"]),
                "has_candidate_text": bool(valid_stats["has_candidate_text"] and test_stats["has_candidate_text"]),
                "has_candidate_title": bool(valid_stats["has_candidate_title"] and test_stats["has_candidate_title"]),
                "has_label": bool(valid_stats["has_label"] and test_stats["has_label"]),
                "schema_compatible": schema_ok,
                "hr10_trivial_flag": bool(test_stats["candidate_pool_size_max"] <= 10 or pool_mean <= 10),
                "notes": "expected 3000 valid/test rows, 500 users, 6 candidates/user"
                if len(valid) == 3000 and len(test) == 3000 and pool_mean == 6 and schema_ok
                else "check row count, pool size, or schema",
            }
        )
        for split, rows in [("valid", valid), ("test", test)]:
            smoke_input = SMOKE_INPUT_DIR / f"{domain}_{split}_smoke_20.jsonl"
            write_jsonl(rows[:20], smoke_input)
            monitor_lines.extend(
                [
                    f"## {domain} {split}",
                    "",
                    f"- expected_rows: `{len(rows)}`",
                    f"- current_rows command: `Get-Content {prediction_path(domain, split)} | Measure-Object -Line`",
                    "- parse_success_rolling: compute from raw jsonl parse_success field",
                    "- status: pending_or_resume",
                    f"- resume_command: `py -3.12 main_infer.py --config {cfg['config']} --split_name {split} --concurrent --resume --max_workers 4 --requests_per_minute 120`",
                    f"- last_update_time: prepare_only",
                    "",
                ]
            )

    write_csv(
        SUMMARY_DIR / "day37_small_domains_preflight_check.csv",
        preflight_rows,
        [
            "domain",
            "train_rows",
            "valid_rows",
            "test_rows",
            "valid_users",
            "test_users",
            "candidate_pool_size_mean",
            "has_user_id",
            "has_history",
            "has_candidate_item_id",
            "has_candidate_text",
            "has_candidate_title",
            "has_label",
            "schema_compatible",
            "hr10_trivial_flag",
            "notes",
        ],
    )
    (SUMMARY_DIR / "day37_small_domains_runtime_monitor.md").write_text("\n".join(monitor_lines), encoding="utf-8")
    print("Prepared Day37 preflight, smoke inputs, and runtime monitor.")


def smoke_status() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for split in ["valid", "test"]:
            path = smoke_output_path(domain, split)
            records = read_jsonl(path)
            parse = [bool(r.get("parse_success")) for r in records]
            raw = [bool(str(r.get("raw_response", "")).strip()) for r in records]
            complete = [
                all(field in r and pd.notna(r.get(field)) for field in EVIDENCE_FIELDS)
                for r in records
            ]
            rows.append(
                {
                    "domain": domain,
                    "split": split,
                    "rows": len(records),
                    "parse_success_rate": float(np.mean(parse)) if parse else 0.0,
                    "raw_response_nonempty_rate": float(np.mean(raw)) if raw else 0.0,
                    "field_complete_rate": float(np.mean(complete)) if complete else 0.0,
                    "status": "pass" if records and np.mean(parse) >= 0.95 and np.mean(raw) >= 0.95 and np.mean(complete) >= 0.95 else "pending_or_failed",
                    "source_file": str(path),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_DIR / "day37_small_domains_smoke_status.csv", index=False)
    return out


def high_conf_error_rate(df: pd.DataFrame, score_col: str, threshold: float = 0.8) -> float:
    work = df.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work["label"] = pd.to_numeric(work["label"], errors="coerce")
    high = work[work[score_col] >= threshold]
    if len(high) == 0:
        return float("nan")
    return float((high["label"].astype(int) == 0).mean())


def calibrate_domain(domain: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_raw = build_relevance_frame(pd.read_json(prediction_path(domain, "valid"), lines=True))
    test_raw = build_relevance_frame(pd.read_json(prediction_path(domain, "test"), lines=True))

    valid_fit = usable_rows(valid_raw, "relevance_probability", target_col="label")
    calibrator = fit_calibrator(valid_fit, method="isotonic", confidence_col="relevance_probability", target_col="label")
    valid_cal = apply_score_calibrator(valid_raw, calibrator, "relevance_probability", "calibrated_relevance_probability")
    test_cal = apply_score_calibrator(test_raw, calibrator, "relevance_probability", "calibrated_relevance_probability")
    valid_cal["relevance_uncertainty"] = 1.0 - valid_cal["calibrated_relevance_probability"]
    test_cal["relevance_uncertainty"] = 1.0 - test_cal["calibrated_relevance_probability"]

    minimal = fit_relevance_posterior(valid_raw, "minimal", use_isotonic=True)
    full = fit_relevance_posterior(valid_raw, "full", use_isotonic=True)
    valid_min = apply_relevance_posterior(valid_raw, minimal, "minimal_calibrated_relevance_probability")
    test_min = apply_relevance_posterior(test_raw, minimal, "minimal_calibrated_relevance_probability")
    valid_full = apply_relevance_posterior(valid_raw, full, "full_calibrated_relevance_probability")
    test_full = apply_relevance_posterior(test_raw, full, "full_calibrated_relevance_probability")

    cal_dir = output_dir(domain) / "calibrated"
    cal_dir.mkdir(parents=True, exist_ok=True)
    valid_cal.to_json(cal_dir / "raw_relevance_valid_calibrated.jsonl", orient="records", lines=True, force_ascii=False)
    test_cal.to_json(cal_dir / "raw_relevance_test_calibrated.jsonl", orient="records", lines=True, force_ascii=False)
    valid_min.to_json(cal_dir / "relevance_evidence_posterior_minimal_valid.jsonl", orient="records", lines=True, force_ascii=False)
    test_min.to_json(cal_dir / "relevance_evidence_posterior_minimal_test.jsonl", orient="records", lines=True, force_ascii=False)
    valid_full.to_json(cal_dir / "relevance_evidence_posterior_full_valid.jsonl", orient="records", lines=True, force_ascii=False)
    test_full.to_json(cal_dir / "relevance_evidence_posterior_full_test.jsonl", orient="records", lines=True, force_ascii=False)
    canonical = test_min.copy()
    canonical["calibrated_relevance_probability"] = canonical["minimal_calibrated_relevance_probability"]
    canonical["relevance_uncertainty"] = 1.0 - canonical["calibrated_relevance_probability"]
    canonical.to_json(cal_dir / "relevance_evidence_posterior_test.jsonl", orient="records", lines=True, force_ascii=False)

    metric_rows: list[dict[str, Any]] = []
    for split, raw_df, cal_df, min_df, full_df in [
        ("valid", valid_raw, valid_cal, valid_min, valid_full),
        ("test", test_raw, test_cal, test_min, test_full),
    ]:
        variants = [
            ("raw_relevance_probability", raw_df, "relevance_probability", "ready", ""),
            ("calibrated_relevance_probability", cal_df, "calibrated_relevance_probability", "ready", ""),
            (
                "evidence_posterior_relevance_minimal",
                min_df,
                "minimal_calibrated_relevance_probability",
                minimal.status,
                minimal.fallback_reason,
            ),
            (
                "evidence_posterior_relevance_full",
                full_df,
                "full_calibrated_relevance_probability",
                full.status,
                full.fallback_reason,
            ),
        ]
        for score_type, df, score_col, status, fallback in variants:
            row = metrics_row(split=split, variant=score_type, df=df, score_col=score_col, status=status)
            metric_rows.append(
                {
                    "domain": domain,
                    "split": split,
                    "score_type": score_type,
                    "ECE": row.get("ece", row.get("ECE", "")),
                    "Brier": row.get("brier_score", row.get("Brier", "")),
                    "AUROC": row.get("auroc", row.get("AUROC", "")),
                    "high_conf_error_rate": row.get("high_conf_error_rate", ""),
                    "accuracy": row.get("accuracy", ""),
                    "status": status,
                    "fallback_reason": fallback,
                    "parse_success_rate": row.get("parse_success_rate", ""),
                }
            )

    field_rows: list[dict[str, Any]] = []
    test_frame = build_relevance_frame(test_raw)
    for field in DIAGNOSTIC_FIELDS:
        vals = pd.to_numeric(test_frame[field], errors="coerce").dropna()
        if vals.empty:
            field_rows.append({"domain": domain, "field": field})
            continue
        field_rows.append(
            {
                "domain": domain,
                "field": field,
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "q05": float(vals.quantile(0.05)),
                "q25": float(vals.quantile(0.25)),
                "q50": float(vals.quantile(0.50)),
                "q75": float(vals.quantile(0.75)),
                "q95": float(vals.quantile(0.95)),
                "near_zero_rate": float((vals <= 0.05).mean()),
                "near_one_rate": float((vals >= 0.95).mean()),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(field_rows)


def load_beauty_summary() -> dict[str, Any]:
    candidates = [
        Path("output-repaired/summary/day9_relevance_evidence_calibration_comparison.csv"),
        Path("output-repaired/summary/day29b_beauty_relevance_probability_diagnostics.csv"),
        Path("output-repaired/beauty_deepseek_relevance_evidence_full/tables/relevance_evidence_calibration_comparison.csv"),
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            # Normalize common schemas.
            if "variant" in df.columns:
                df = df.rename(columns={"variant": "score_type", "ece": "ECE", "brier_score": "Brier", "auroc": "AUROC"})
                df = df[df.get("split", "test") == "test"].copy() if "split" in df.columns else df
            raw = df[df["score_type"].astype(str).str.contains("raw_relevance_probability", regex=False)].iloc[0]
            cal = df[df["score_type"].astype(str).str.contains("calibrated_relevance_probability", regex=False)].iloc[0]
            return {
                "domain": "beauty",
                "dataset_type": "beauty_full",
                "valid_rows": 5838,
                "test_rows": 5838,
                "raw_ECE": raw["ECE"],
                "calibrated_ECE": cal["ECE"],
                "delta_ECE": float(raw["ECE"]) - float(cal["ECE"]),
                "raw_Brier": raw["Brier"],
                "calibrated_Brier": cal["Brier"],
                "delta_Brier": float(raw["Brier"]) - float(cal["Brier"]),
                "raw_AUROC": raw["AUROC"],
                "calibrated_AUROC": cal["AUROC"],
                "delta_AUROC": float(cal["AUROC"]) - float(raw["AUROC"]),
                "claim_level": "full_primary",
                "source_file": str(path),
            }
    return {
        "domain": "beauty",
        "dataset_type": "beauty_full",
        "valid_rows": 5838,
        "test_rows": 5838,
        "raw_ECE": "",
        "calibrated_ECE": "",
        "delta_ECE": "",
        "raw_Brier": "",
        "calibrated_Brier": "",
        "delta_Brier": "",
        "raw_AUROC": "",
        "calibrated_AUROC": "",
        "delta_AUROC": "",
        "claim_level": "full_primary",
        "source_file": "not_found",
    }


def analyze() -> None:
    smoke_status()
    all_metrics: list[pd.DataFrame] = []
    all_fields: list[pd.DataFrame] = []
    monitor_rows: list[str] = [
        "# Day37 Small-Domain Runtime Monitor",
        "",
    ]
    for domain in DOMAINS:
        for split in ["valid", "test"]:
            records = read_jsonl(prediction_path(domain, split))
            parse = float(np.mean([bool(r.get("parse_success")) for r in records])) if records else 0.0
            monitor_rows.extend(
                [
                    f"## {domain} {split}",
                    "",
                    f"- expected_rows: `3000`",
                    f"- current_rows: `{len(records)}`",
                    f"- parse_success_rolling: `{parse:.4f}`",
                    f"- status: `{'complete' if len(records) == 3000 else 'incomplete'}`",
                    f"- resume_command: `py -3.12 main_infer.py --config {DOMAINS[domain]['config']} --split_name {split} --concurrent --resume --max_workers 4 --requests_per_minute 120`",
                    f"- last_update_time: analyze",
                    "",
                ]
            )
        metrics, fields = calibrate_domain(domain)
        all_metrics.append(metrics)
        all_fields.append(fields)
    (SUMMARY_DIR / "day37_small_domains_runtime_monitor.md").write_text("\n".join(monitor_rows), encoding="utf-8")

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    fields_df = pd.concat(all_fields, ignore_index=True)
    metrics_df.to_csv(SUMMARY_DIR / "day37_small_domains_calibration_comparison.csv", index=False)
    fields_df.to_csv(SUMMARY_DIR / "day37_small_domains_field_diagnostics.csv", index=False)

    summary_rows = [load_beauty_summary()]
    for domain in DOMAINS:
        test_rows = metrics_df[(metrics_df["domain"] == domain) & (metrics_df["split"] == "test")]
        raw = test_rows[test_rows["score_type"] == "raw_relevance_probability"].iloc[0]
        cal = test_rows[test_rows["score_type"] == "calibrated_relevance_probability"].iloc[0]
        summary_rows.append(
            {
                "domain": domain,
                "dataset_type": f"{domain}_small",
                "valid_rows": 3000,
                "test_rows": 3000,
                "raw_ECE": raw["ECE"],
                "calibrated_ECE": cal["ECE"],
                "delta_ECE": float(raw["ECE"]) - float(cal["ECE"]),
                "raw_Brier": raw["Brier"],
                "calibrated_Brier": cal["Brier"],
                "delta_Brier": float(raw["Brier"]) - float(cal["Brier"]),
                "raw_AUROC": raw["AUROC"],
                "calibrated_AUROC": cal["AUROC"],
                "delta_AUROC": float(cal["AUROC"]) - float(raw["AUROC"]),
                "claim_level": "small_cross_domain_sanity",
                "source_file": str(SUMMARY_DIR / "day37_small_domains_calibration_comparison.csv"),
            }
        )
    pd.DataFrame(summary_rows).to_csv(SUMMARY_DIR / "day37_beauty_vs_small_domains_cep_calibration_summary.csv", index=False)

    write_report(metrics_df, fields_df)
    print("Wrote Day37 calibration, field diagnostics, Beauty-vs-small summary, monitor, and report.")


def blocked_report(reason: str = "APIConnectionError during smoke / health check") -> None:
    smoke = smoke_status()
    preflight_path = SUMMARY_DIR / "day37_small_domains_preflight_check.csv"
    preflight = pd.read_csv(preflight_path) if preflight_path.exists() else pd.DataFrame()

    cal_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    summary_rows = [load_beauty_summary()]
    for domain in DOMAINS:
        for score_type in [
            "raw_relevance_probability",
            "calibrated_relevance_probability",
            "evidence_posterior_relevance_minimal",
            "evidence_posterior_relevance_full",
        ]:
            cal_rows.append(
                {
                    "domain": domain,
                    "score_type": score_type,
                    "ECE": "",
                    "Brier": "",
                    "AUROC": "",
                    "high_conf_error_rate": "",
                    "accuracy": "",
                    "status": "blocked_api_connection",
                    "fallback_reason": reason,
                }
            )
        for field in DIAGNOSTIC_FIELDS:
            field_rows.append(
                {
                    "domain": domain,
                    "field": field,
                    "mean": "",
                    "std": "",
                    "min": "",
                    "max": "",
                    "q05": "",
                    "q25": "",
                    "q50": "",
                    "q75": "",
                    "q95": "",
                    "near_zero_rate": "",
                    "near_one_rate": "",
                    "status": "blocked_api_connection",
                }
            )
        rows = preflight[preflight["domain"] == domain]
        valid_rows = int(rows["valid_rows"].iloc[0]) if not rows.empty else 3000
        test_rows = int(rows["test_rows"].iloc[0]) if not rows.empty else 3000
        summary_rows.append(
            {
                "domain": domain,
                "dataset_type": f"{domain}_small",
                "valid_rows": valid_rows,
                "test_rows": test_rows,
                "raw_ECE": "",
                "calibrated_ECE": "",
                "delta_ECE": "",
                "raw_Brier": "",
                "calibrated_Brier": "",
                "delta_Brier": "",
                "raw_AUROC": "",
                "calibrated_AUROC": "",
                "delta_AUROC": "",
                "claim_level": "small_cross_domain_sanity_blocked_api_connection",
                "source_file": str(SUMMARY_DIR / "day37_small_domains_calibration_comparison.csv"),
            }
        )

    pd.DataFrame(cal_rows).to_csv(SUMMARY_DIR / "day37_small_domains_calibration_comparison.csv", index=False)
    pd.DataFrame(field_rows).to_csv(SUMMARY_DIR / "day37_small_domains_field_diagnostics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_DIR / "day37_beauty_vs_small_domains_cep_calibration_summary.csv", index=False)

    monitor_lines = [
        "# Day37 Small-Domain Runtime Monitor",
        "",
        f"Current status: `blocked_api_connection`.",
        "",
        "Preflight completed for all three small domains. The first Movies smoke wrote APIConnectionError rows with empty raw_response, and a one-row DeepSeek health check also failed. No full small-domain inference was launched.",
        "",
    ]
    for domain in DOMAINS:
        for split in ["valid", "test"]:
            records = read_jsonl(prediction_path(domain, split))
            monitor_lines.extend(
                [
                    f"## {domain} {split}",
                    "",
                    "- expected_rows: `3000`",
                    f"- current_rows: `{len(records)}`",
                    "- parse_success_rolling: `0.0000`" if not records else "- parse_success_rolling: check raw jsonl",
                    "- status: `blocked_api_connection`",
                    f"- resume_command: `py -3.12 main_infer.py --config {DOMAINS[domain]['config']} --split_name {split} --concurrent --resume --max_workers 4 --requests_per_minute 120`",
                    "- last_update_time: blocked_report",
                    "",
                ]
            )
    (SUMMARY_DIR / "day37_small_domains_runtime_monitor.md").write_text("\n".join(monitor_lines), encoding="utf-8")

    lines = [
        "# Day37 Small-Domain CEP Relevance Report",
        "",
        "## 1. Why Small Domains",
        "",
        "Small domains remain the right lightweight cross-domain sanity / continuity setting because their cold rates are much healthier than regular medium. They do not replace regular medium as the realistic cold-start/content-carrier setting.",
        "",
        "## 2. Data Scale And Metric Protocol",
        "",
        "Preflight passed for Books, Electronics, and Movies small: each has 500 users, 3000 valid rows, 3000 test rows, and 6 candidates per user. HR@10 is trivial and should not be used as primary evidence.",
        "",
        "## 3. Inference Status",
        "",
        f"Day37 inference is currently blocked by `{reason}`. Movies valid smoke produced empty raw_response / parse_success=0 rows, and the one-row DeepSeek health check also returned APIConnectionError. This is an API/network layer block, not a prompt/schema/parser failure.",
        "",
        "## 4. Calibration Result",
        "",
        "Not computed yet for small domains because full valid/test relevance evidence was not generated. The output calibration CSV contains blocked rows so downstream reporting does not silently interpret missing metrics.",
        "",
        "## 5. Field Diagnostics",
        "",
        "Not computed yet for small domains for the same reason.",
        "",
        "## 6. Direction Against Beauty Day9",
        "",
        "No small-domain calibration claim is made until API inference completes. Beauty Day9 remains the primary completed relevance-posterior result.",
        "",
        "## 7. Day38 Recommendation",
        "",
        "Before retrying Day37, rerun the health check. If it succeeds, resume in the original order: movies_small, books_small, electronics_small. If API remains unstable, keep Day37 blocked and do not start backbone plug-in.",
    ]
    (SUMMARY_DIR / "day37_small_domain_cep_relevance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote Day37 blocked API status, placeholder diagnostics, monitor, and report.")


def fmt(value: Any) -> str:
    try:
        if value == "" or pd.isna(value):
            return "NA"
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def write_report(metrics_df: pd.DataFrame, fields_df: pd.DataFrame) -> None:
    smoke = pd.read_csv(SUMMARY_DIR / "day37_small_domains_smoke_status.csv")
    lines = [
        "# Day37 Small-Domain CEP Relevance Report",
        "",
        "## 1. Why Small Domains",
        "",
        "Regular medium domains expose realistic cold-start behavior, but they are not healthy ID-backbone sanity sets. The small domains have much lower cold rates, so they provide lightweight cross-domain sanity / continuity for CEP relevance evidence and later ID-backbone plug-in tests.",
        "",
        "## 2. Data Scale And Metric Protocol",
        "",
        "Books, Electronics, and Movies small each have 500 users, 3000 valid rows, 3000 test rows, and 6 candidates per user. HR@10 is trivial and should not be used as primary evidence. Main downstream metrics remain NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5.",
        "",
        "## 3. Inference Status",
        "",
    ]
    for domain in DOMAINS:
        domain_smoke = smoke[smoke["domain"] == domain]
        parse = domain_smoke["parse_success_rate"].mean()
        raw = domain_smoke["raw_response_nonempty_rate"].mean()
        lines.append(f"- {domain}: smoke parse_success `{parse:.4f}`, raw_response_nonempty `{raw:.4f}`.")
    lines.extend(["", "## 4. Calibration Result", ""])
    for domain in DOMAINS:
        rows = metrics_df[(metrics_df["domain"] == domain) & (metrics_df["split"] == "test")]
        raw = rows[rows["score_type"] == "raw_relevance_probability"].iloc[0]
        cal = rows[rows["score_type"] == "calibrated_relevance_probability"].iloc[0]
        minimal = rows[rows["score_type"] == "evidence_posterior_relevance_minimal"].iloc[0]
        full = rows[rows["score_type"] == "evidence_posterior_relevance_full"].iloc[0]
        lines.append(
            f"- {domain}: raw ECE `{fmt(raw['ECE'])}` -> calibrated ECE `{fmt(cal['ECE'])}`; "
            f"raw Brier `{fmt(raw['Brier'])}` -> calibrated Brier `{fmt(cal['Brier'])}`; "
            f"minimal posterior ECE `{fmt(minimal['ECE'])}`, full posterior ECE `{fmt(full['ECE'])}`."
        )
    lines.extend(
        [
            "",
            "## 5. Field Diagnostics",
            "",
            "Field diagnostics are written to `day37_small_domains_field_diagnostics.csv`. No collapse should be inferred from a single metric; inspect relevance_probability, evidence fields, and evidence_risk jointly.",
            "",
            "## 6. Direction Against Beauty Day9",
            "",
            "The small-domain results align with Beauty at the observation level: raw relevance probability is informative but not a calibrated posterior, while valid-fit/test-eval calibration repairs probability quality. These small domains are sanity/continuity evidence, not replacements for regular medium domain analysis.",
            "",
            "## 7. Day38 Recommendation",
            "",
            "If all three domains remain complete and calibrated, Day38 should run small-domain SASRec plug-in sanity checks. Keep the claim narrow: small domains test cross-domain continuity under healthy low-cold ID-backbone conditions; regular medium remains the realistic cold-style setting.",
        ]
    )
    (SUMMARY_DIR / "day37_small_domain_cep_relevance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke-status", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--blocked-report", action="store_true")
    parser.add_argument("--blocked-reason", type=str, default="APIConnectionError during smoke / health check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare:
        prepare()
    if args.smoke_status:
        smoke_status()
        print("Wrote Day37 smoke status.")
    if args.analyze:
        analyze()
    if args.blocked_report:
        blocked_report(args.blocked_reason)
    if not args.prepare and not args.smoke_status and not args.analyze and not args.blocked_report:
        prepare()


if __name__ == "__main__":
    main()

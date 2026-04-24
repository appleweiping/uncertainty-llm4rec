from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.paths import ensure_compare_dirs


DEFAULT_METRICS = [
    "num_eval_samples",
    "parse_success_rate",
    "high_conf_error_rate",
    "calib_accuracy",
    "calib_avg_confidence",
    "calib_brier_score",
    "calib_ece",
    "calib_auroc",
    "rank_HR@10",
    "rank_NDCG@10",
    "rank_MRR@10",
    "rank_head_exposure_ratio@10",
    "rank_tail_exposure_ratio@10",
    "rank_long_tail_coverage@10",
    "rerank_HR@10",
    "rerank_NDCG@10",
    "rerank_MRR@10",
    "rerank_head_exposure_ratio@10",
    "rerank_tail_exposure_ratio@10",
    "rerank_long_tail_coverage@10",
]

LOWER_IS_BETTER_HINTS = [
    "ece",
    "brier",
    "error",
    "uncertainty",
]


def read_estimator_table(exp_name: str, output_root: str | Path) -> pd.DataFrame:
    path = Path(output_root) / exp_name / "tables" / "estimator_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Estimator comparison not found for {exp_name}: {path}. "
            "Run main_uncertainty_compare.py first."
        )
    return pd.read_csv(path)


def metric_direction(metric: str) -> str:
    normalized = metric.lower()
    if any(token in normalized for token in LOWER_IS_BETTER_HINTS):
        return "lower_is_better"
    return "higher_is_better"


def judge_delta(metric: str, delta: float) -> str:
    if pd.isna(delta):
        return "missing"
    if abs(float(delta)) < 1e-12:
        return "unchanged"
    direction = metric_direction(metric)
    if direction == "lower_is_better":
        return "improved" if delta < 0 else "degraded"
    return "improved" if delta > 0 else "degraded"


def build_robustness_summary(
    clean_df: pd.DataFrame,
    noisy_df: pd.DataFrame,
    *,
    metrics: list[str],
) -> pd.DataFrame:
    shared_estimators = sorted(set(clean_df["estimator"]) & set(noisy_df["estimator"]))
    rows: list[dict] = []

    for estimator in shared_estimators:
        clean_row = clean_df[clean_df["estimator"] == estimator].iloc[0]
        noisy_row = noisy_df[noisy_df["estimator"] == estimator].iloc[0]

        for metric in metrics:
            if metric not in clean_df.columns or metric not in noisy_df.columns:
                continue
            clean_value = pd.to_numeric(pd.Series([clean_row[metric]]), errors="coerce").iloc[0]
            noisy_value = pd.to_numeric(pd.Series([noisy_row[metric]]), errors="coerce").iloc[0]
            delta = noisy_value - clean_value
            relative_delta = delta / abs(clean_value) if pd.notna(clean_value) and abs(clean_value) > 1e-12 else pd.NA

            rows.append(
                {
                    "estimator": estimator,
                    "metric": metric,
                    "metric_direction": metric_direction(metric),
                    "clean_value": clean_value,
                    "noisy_value": noisy_value,
                    "delta_noisy_minus_clean": delta,
                    "relative_delta": relative_delta,
                    "robustness_judgment": judge_delta(metric, delta),
                }
            )

    return pd.DataFrame(rows)


def build_markdown(summary_df: pd.DataFrame, clean_exp_name: str, noisy_exp_name: str) -> str:
    lines = [
        f"# Evidence Robustness: {clean_exp_name} vs {noisy_exp_name}",
        "",
        "This table compares estimator behavior under clean and noisy inputs. "
        "Negative deltas are desirable for ECE, Brier, and high-confidence error rate; "
        "positive deltas are desirable for AUROC and ranking metrics.",
        "",
    ]

    if summary_df.empty:
        lines.append("No shared estimator metrics were available.")
        return "\n".join(lines) + "\n"

    focus_metrics = [
        "high_conf_error_rate",
        "calib_brier_score",
        "calib_ece",
        "calib_auroc",
        "rerank_NDCG@10",
    ]
    focus = summary_df[summary_df["metric"].isin(focus_metrics)].copy()
    if focus.empty:
        focus = summary_df.head(20).copy()

    columns = ["estimator", "metric", "clean_value", "noisy_value", "delta_noisy_minus_clean", "robustness_judgment"]
    lines.append("| estimator | metric | clean | noisy | delta | judgment |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for _, row in focus[columns].iterrows():
        lines.append(
            "| {estimator} | {metric} | {clean:.6g} | {noisy:.6g} | {delta:.6g} | {judgment} |".format(
                estimator=row["estimator"],
                metric=row["metric"],
                clean=float(row["clean_value"]) if pd.notna(row["clean_value"]) else float("nan"),
                noisy=float(row["noisy_value"]) if pd.notna(row["noisy_value"]) else float("nan"),
                delta=float(row["delta_noisy_minus_clean"]) if pd.notna(row["delta_noisy_minus_clean"]) else float("nan"),
                judgment=row["robustness_judgment"],
            )
        )
    lines.append("")
    lines.append(
        "Interpretation should stay conservative: this first-pass robustness view checks whether "
        "evidence posterior confidence degrades more gracefully than raw confidence under controlled noise."
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_exp_name", type=str, required=True)
    parser.add_argument("--noisy_exp_name", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="output-repaired")
    parser.add_argument("--compare_name", type=str, default=None)
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit metric columns to compare. Defaults to evidence/ranking robustness metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_df = read_estimator_table(args.clean_exp_name, args.output_root)
    noisy_df = read_estimator_table(args.noisy_exp_name, args.output_root)

    metrics = args.metrics if args.metrics is not None else DEFAULT_METRICS
    summary_df = build_robustness_summary(clean_df, noisy_df, metrics=metrics)

    compare_name = args.compare_name or f"{args.clean_exp_name}_vs_{args.noisy_exp_name}"
    compare_root = ensure_compare_dirs(compare_name, args.output_root)
    table_path = compare_root / "tables" / "evidence_robustness_summary.csv"
    md_path = compare_root / "tables" / "evidence_robustness_summary.md"

    summary_df.to_csv(table_path, index=False)
    md_path.write_text(
        build_markdown(summary_df, args.clean_exp_name, args.noisy_exp_name),
        encoding="utf-8",
    )

    print(f"Saved evidence robustness summary to: {table_path}")
    print(f"Saved evidence robustness markdown to: {md_path}")


if __name__ == "__main__":
    main()

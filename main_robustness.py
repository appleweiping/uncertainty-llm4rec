# main_robustness.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.noise_analysis import summarize_noise_effect
from src.eval.robustness_metrics import build_robustness_table, build_scalar_robustness_table
from src.utils.paths import ensure_compare_dirs


def _read_single_row_csv(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {str(k): float(v) for k, v in row.items() if k != "num_samples"}


def _read_test_calibration_metrics(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    if df.empty:
        return {}

    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        return {}

    metrics: dict[str, float] = {}
    for _, row in test_df.iterrows():
        metric = str(row["metric"])
        metrics[f"{metric}_before"] = float(row["before"])
        metrics[f"{metric}_after"] = float(row["after"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean_exp",
        type=str,
        default="clean",
        help="Experiment name for clean setting."
    )
    parser.add_argument(
        "--noisy_exp",
        type=str,
        default="noisy",
        help="Experiment name for noisy setting."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs",
        help="Root directory for all experiment outputs."
    )
    parser.add_argument(
        "--results_filename",
        type=str,
        default="rerank_results.csv",
        help="Filename under each experiment's tables/ directory."
    )
    parser.add_argument(
        "--diagnostic_filename",
        type=str,
        default="diagnostic_metrics.csv",
        help="Filename for raw diagnostic metrics under tables/.",
    )
    parser.add_argument(
        "--calibration_filename",
        type=str,
        default="calibration_comparison.csv",
        help="Filename for calibration comparison under tables/.",
    )
    parser.add_argument(
        "--confidence_filename",
        type=str,
        default="confidence_correctness_summary.csv",
        help="Filename for confidence/correctness summary under tables/.",
    )
    args = parser.parse_args()

    clean_path = Path(args.output_root) / args.clean_exp / "tables" / args.results_filename
    noisy_path = Path(args.output_root) / args.noisy_exp / "tables" / args.results_filename
    clean_diagnostic_path = Path(args.output_root) / args.clean_exp / "tables" / args.diagnostic_filename
    noisy_diagnostic_path = Path(args.output_root) / args.noisy_exp / "tables" / args.diagnostic_filename
    clean_calibration_path = Path(args.output_root) / args.clean_exp / "tables" / args.calibration_filename
    noisy_calibration_path = Path(args.output_root) / args.noisy_exp / "tables" / args.calibration_filename
    clean_confidence_path = Path(args.output_root) / args.clean_exp / "tables" / args.confidence_filename
    noisy_confidence_path = Path(args.output_root) / args.noisy_exp / "tables" / args.confidence_filename

    if not clean_path.exists():
        raise FileNotFoundError(f"Clean result file not found: {clean_path}")
    if not noisy_path.exists():
        raise FileNotFoundError(f"Noisy result file not found: {noisy_path}")
    for path in [
        clean_diagnostic_path,
        noisy_diagnostic_path,
        clean_calibration_path,
        noisy_calibration_path,
        clean_confidence_path,
        noisy_confidence_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required robustness input file not found: {path}")

    clean_df = pd.read_csv(clean_path)
    noisy_df = pd.read_csv(noisy_path)

    robustness_df = build_robustness_table(clean_df, noisy_df)
    diagnostic_df = build_scalar_robustness_table(
        _read_single_row_csv(clean_diagnostic_path),
        _read_single_row_csv(noisy_diagnostic_path),
    )
    calibration_df = build_scalar_robustness_table(
        _read_test_calibration_metrics(clean_calibration_path),
        _read_test_calibration_metrics(noisy_calibration_path),
    )
    confidence_df = build_scalar_robustness_table(
        _read_single_row_csv(clean_confidence_path),
        _read_single_row_csv(noisy_confidence_path),
    )

    compare_name = f"{args.clean_exp}_vs_{args.noisy_exp}"
    compare_root = ensure_compare_dirs(compare_name, args.output_root)
    tables_dir = compare_root / "tables"

    robustness_table_path = tables_dir / "robustness_table.csv"
    diagnostic_table_path = tables_dir / "robustness_diagnostic_table.csv"
    calibration_table_path = tables_dir / "robustness_calibration_table.csv"
    confidence_table_path = tables_dir / "robustness_confidence_table.csv"
    robustness_summary_path = tables_dir / "robustness_summary.csv"

    robustness_df.to_csv(robustness_table_path, index=False)
    diagnostic_df.to_csv(diagnostic_table_path, index=False)
    calibration_df.to_csv(calibration_table_path, index=False)
    confidence_df.to_csv(confidence_table_path, index=False)

    summary = {}
    for prefix, df in [
        ("rerank", robustness_df),
        ("diagnostic", diagnostic_df),
        ("calibration", calibration_df),
        ("confidence", confidence_df),
    ]:
        df_summary = summarize_noise_effect(df)
        for key, value in df_summary.items():
            summary[f"{prefix}_{key}"] = value
    pd.DataFrame([summary]).to_csv(robustness_summary_path, index=False)

    print(f"Saved robustness table to:   {robustness_table_path}")
    print(f"Saved diagnostic robustness table to: {diagnostic_table_path}")
    print(f"Saved calibration robustness table to: {calibration_table_path}")
    print(f"Saved confidence robustness table to: {confidence_table_path}")
    print(f"Saved robustness summary to: {robustness_summary_path}")


if __name__ == "__main__":
    main()

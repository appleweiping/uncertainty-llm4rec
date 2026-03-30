# main_robustness.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.noise_analysis import summarize_noise_effect
from src.eval.robustness_metrics import build_robustness_table
from src.utils.paths import ensure_compare_dirs


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
    args = parser.parse_args()

    clean_path = Path(args.output_root) / args.clean_exp / "tables" / args.results_filename
    noisy_path = Path(args.output_root) / args.noisy_exp / "tables" / args.results_filename

    if not clean_path.exists():
        raise FileNotFoundError(f"Clean result file not found: {clean_path}")
    if not noisy_path.exists():
        raise FileNotFoundError(f"Noisy result file not found: {noisy_path}")

    clean_df = pd.read_csv(clean_path)
    noisy_df = pd.read_csv(noisy_path)

    robustness_df = build_robustness_table(clean_df, noisy_df)

    compare_name = f"{args.clean_exp}_vs_{args.noisy_exp}"
    compare_root = ensure_compare_dirs(compare_name, args.output_root)
    tables_dir = compare_root / "tables"

    robustness_table_path = tables_dir / "robustness_table.csv"
    robustness_summary_path = tables_dir / "robustness_summary.csv"

    robustness_df.to_csv(robustness_table_path, index=False)

    summary = summarize_noise_effect(robustness_df)
    pd.DataFrame([summary]).to_csv(robustness_summary_path, index=False)

    print(f"Saved robustness table to:   {robustness_table_path}")
    print(f"Saved robustness summary to: {robustness_summary_path}")


if __name__ == "__main__":
    main()
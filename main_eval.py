# main_eval.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.confidence_correctness import (
    compute_confidence_bins_accuracy,
    compute_confidence_correctness_summary,
    prepare_prediction_dataframe,
)
from src.analysis.exposure_analysis import compute_high_confidence_exposure
from src.analysis.plotting import (
    plot_confidence_histogram,
    plot_high_confidence_exposure_shift,
    plot_popularity_avg_confidence,
    plot_popularity_confidence_boxplot,
    plot_reliability_diagram,
)
from src.analysis.popularity_bias import compute_popularity_group_stats
from src.eval.calibration_metrics import (
    compute_calibration_metrics,
    get_reliability_dataframe,
)
from src.utils.paths import ensure_exp_dirs


def load_jsonl(path: str | Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_summary_dict(summary: dict, path: str | Path) -> None:
    df = pd.DataFrame([summary])
    save_table(df, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="clean",
        help="Experiment name."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Optional explicit path to prediction jsonl. Defaults to outputs/{exp_name}/predictions/test_raw.jsonl"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs",
        help="Root directory for all experiment outputs."
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=10,
        help="Number of bins for reliability analysis."
    )
    parser.add_argument(
        "--high_conf_threshold",
        type=float,
        default=0.8,
        help="Threshold for high-confidence analysis."
    )
    args = parser.parse_args()

    paths = ensure_exp_dirs(args.exp_name, args.output_root)
    input_path = (
        Path(args.input_path)
        if args.input_path is not None
        else paths.predictions_dir / "test_raw.jsonl"
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {input_path}")

    print(f"[{args.exp_name}] Loading predictions from: {input_path}")
    raw_df = load_jsonl(input_path)
    df = prepare_prediction_dataframe(raw_df)

    print(f"[{args.exp_name}] Loaded {len(df)} samples.")

    metrics = compute_calibration_metrics(df, confidence_col="confidence", n_bins=args.n_bins)
    save_summary_dict(metrics, paths.tables_dir / "diagnostic_metrics.csv")

    cc_summary = compute_confidence_correctness_summary(
        df,
        high_conf_threshold=args.high_conf_threshold
    )
    save_summary_dict(cc_summary, paths.tables_dir / "confidence_correctness_summary.csv")

    bins_df = compute_confidence_bins_accuracy(df, n_bins=args.n_bins)
    save_table(bins_df, paths.tables_dir / "confidence_bins_accuracy.csv")

    reliability_df = get_reliability_dataframe(
        df["label"].to_numpy(),
        df["confidence"].to_numpy(),
        n_bins=args.n_bins
    )
    save_table(reliability_df, paths.tables_dir / "reliability_bins.csv")

    pop_df = compute_popularity_group_stats(
        df,
        high_conf_threshold=args.high_conf_threshold
    )
    save_table(pop_df, paths.tables_dir / "popularity_group_stats.csv")

    exposure_df = compute_high_confidence_exposure(
        df,
        high_conf_threshold=args.high_conf_threshold
    )
    save_table(exposure_df, paths.tables_dir / "high_confidence_exposure.csv")

    plot_confidence_histogram(df, paths.figures_dir / "confidence_histogram_correct_vs_wrong.png")
    plot_reliability_diagram(reliability_df, paths.figures_dir / "reliability_diagram.png")
    plot_popularity_avg_confidence(pop_df, paths.figures_dir / "popularity_avg_confidence.png")
    plot_popularity_confidence_boxplot(df, paths.figures_dir / "popularity_confidence_boxplot.png")
    plot_high_confidence_exposure_shift(exposure_df, paths.figures_dir / "high_confidence_exposure_shift.png")

    print(f"[{args.exp_name}] Evaluation done.")
    print(f"Tables saved to:  {paths.tables_dir}")
    print(f"Figures saved to: {paths.figures_dir}")


if __name__ == "__main__":
    main()
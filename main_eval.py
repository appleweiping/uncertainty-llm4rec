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
        "--input_path",
        type=str,
        default="outputs/predictions/test_raw.jsonl",
        help="Path to raw prediction jsonl file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Base output directory."
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

    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from: {args.input_path}")
    raw_df = load_jsonl(args.input_path)
    df = prepare_prediction_dataframe(raw_df)

    print(f"Loaded {len(df)} samples.")

    # 1) overall calibration metrics
    metrics = compute_calibration_metrics(df, confidence_col="confidence", n_bins=args.n_bins)
    save_summary_dict(metrics, tables_dir / "diagnostic_metrics.csv")
    print("Saved diagnostic_metrics.csv")

    # 2) confidence-correctness summary
    cc_summary = compute_confidence_correctness_summary(
        df,
        high_conf_threshold=args.high_conf_threshold
    )
    save_summary_dict(cc_summary, tables_dir / "confidence_correctness_summary.csv")
    print("Saved confidence_correctness_summary.csv")

    # 3) confidence bins accuracy
    bins_df = compute_confidence_bins_accuracy(df, n_bins=args.n_bins)
    save_table(bins_df, tables_dir / "confidence_bins_accuracy.csv")
    print("Saved confidence_bins_accuracy.csv")

    # 4) reliability dataframe
    reliability_df = get_reliability_dataframe(
        df["label"].to_numpy(),
        df["confidence"].to_numpy(),
        n_bins=args.n_bins
    )
    save_table(reliability_df, tables_dir / "reliability_bins.csv")
    print("Saved reliability_bins.csv")

    # 5) popularity stats
    pop_df = compute_popularity_group_stats(
        df,
        high_conf_threshold=args.high_conf_threshold
    )
    save_table(pop_df, tables_dir / "popularity_group_stats.csv")
    print("Saved popularity_group_stats.csv")

    # 6) exposure stats
    exposure_df = compute_high_confidence_exposure(
        df,
        high_conf_threshold=args.high_conf_threshold
    )
    save_table(exposure_df, tables_dir / "high_confidence_exposure.csv")
    print("Saved high_confidence_exposure.csv")

    # 7) figures
    plot_confidence_histogram(df, figures_dir / "confidence_histogram_correct_vs_wrong.png")
    plot_reliability_diagram(reliability_df, figures_dir / "reliability_diagram.png")
    plot_popularity_avg_confidence(pop_df, figures_dir / "popularity_avg_confidence.png")
    plot_popularity_confidence_boxplot(df, figures_dir / "popularity_confidence_boxplot.png")
    plot_high_confidence_exposure_shift(exposure_df, figures_dir / "high_confidence_exposure_shift.png")
    print("Saved figures to outputs/figures/")

    print("\nDone.")
    print("Main outputs:")
    print(f"- {tables_dir / 'diagnostic_metrics.csv'}")
    print(f"- {tables_dir / 'confidence_correctness_summary.csv'}")
    print(f"- {tables_dir / 'confidence_bins_accuracy.csv'}")
    print(f"- {tables_dir / 'reliability_bins.csv'}")
    print(f"- {tables_dir / 'popularity_group_stats.csv'}")
    print(f"- {tables_dir / 'high_confidence_exposure.csv'}")
    print(f"- {figures_dir / 'confidence_histogram_correct_vs_wrong.png'}")
    print(f"- {figures_dir / 'reliability_diagram.png'}")
    print(f"- {figures_dir / 'popularity_avg_confidence.png'}")
    print(f"- {figures_dir / 'popularity_confidence_boxplot.png'}")
    print(f"- {figures_dir / 'high_confidence_exposure_shift.png'}")


if __name__ == "__main__":
    main()
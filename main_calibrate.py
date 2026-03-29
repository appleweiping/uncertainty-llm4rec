# main_calibrate.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.calibration_plotting import plot_before_after_reliability
from src.analysis.confidence_correctness import prepare_prediction_dataframe
from src.eval.calibration_metrics import (
    compute_calibration_metrics,
    get_reliability_dataframe,
)
from src.uncertainty.calibration import (
    apply_calibrator,
    build_split_metadata,
    fit_calibrator,
    user_level_split,
)
from src.uncertainty.verbalized_confidence import (
    add_uncertainty_from_confidence,
    normalize_confidence_column,
)


def load_jsonl(path: str | Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def save_jsonl(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_summary_dict(summary: dict, path: str | Path) -> None:
    save_table(pd.DataFrame([summary]), path)


def compare_metrics(raw_df: pd.DataFrame, calibrated_df: pd.DataFrame) -> pd.DataFrame:
    before_metrics = compute_calibration_metrics(raw_df, confidence_col="confidence", n_bins=10)
    after_metrics = compute_calibration_metrics(calibrated_df, confidence_col="calibrated_confidence", n_bins=10)

    rows = []
    for metric_name in before_metrics.keys():
        rows.append(
            {
                "metric": metric_name,
                "before": before_metrics[metric_name],
                "after": after_metrics[metric_name],
            }
        )
    return pd.DataFrame(rows)


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
        "--method",
        type=str,
        default="isotonic",
        choices=["isotonic", "platt"],
        help="Calibration method."
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.5,
        help="User-level validation split ratio."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for split."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    calibrated_dir = output_dir / "calibrated"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    calibrated_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from: {args.input_path}")
    raw_df = load_jsonl(args.input_path)

    # Step 1: standardize + add is_correct
    df = prepare_prediction_dataframe(raw_df)
    df = normalize_confidence_column(df, input_col="confidence", output_col="confidence")

    print(f"Loaded {len(df)} samples from raw predictions.")

    # Step 2: valid/test split at user level
    split_result = user_level_split(
        df,
        user_col="user_id",
        valid_ratio=args.valid_ratio,
        random_state=args.random_state
    )
    valid_df = split_result.valid_df
    test_df = split_result.test_df

    split_meta = build_split_metadata(valid_df, test_df)
    save_summary_dict(split_meta, tables_dir / "calibration_split_metadata.csv")
    print("Saved calibration_split_metadata.csv")

    # Step 3: fit calibrator on valid
    calibrator = fit_calibrator(
        valid_df=valid_df,
        method=args.method,
        confidence_col="confidence",
        target_col="is_correct"
    )
    print(f"Fitted {args.method} calibrator on valid split.")

    # Step 4: apply on valid and test
    valid_calibrated = apply_calibrator(valid_df, calibrator, input_col="confidence", output_col="calibrated_confidence")
    valid_calibrated = add_uncertainty_from_confidence(valid_calibrated, confidence_col="calibrated_confidence", output_col="uncertainty")

    test_calibrated = apply_calibrator(test_df, calibrator, input_col="confidence", output_col="calibrated_confidence")
    test_calibrated = add_uncertainty_from_confidence(test_calibrated, confidence_col="calibrated_confidence", output_col="uncertainty")

    # Step 5: save calibrated test file
    save_jsonl(test_calibrated, calibrated_dir / "test_calibrated.jsonl")
    print("Saved outputs/calibrated/test_calibrated.jsonl")

    # Optional: also save valid side for debugging
    save_jsonl(valid_calibrated, calibrated_dir / "valid_calibrated.jsonl")
    print("Saved outputs/calibrated/valid_calibrated.jsonl")

    # Step 6: metric comparison on test set
    comparison_df = compare_metrics(test_df, test_calibrated)
    save_table(comparison_df, tables_dir / "calibration_comparison.csv")
    print("Saved calibration_comparison.csv")

    # Step 7: reliability bins before/after
    before_rel = get_reliability_dataframe(
        test_df["label"].to_numpy(),
        test_df["confidence"].to_numpy(),
        n_bins=10
    )
    after_rel = get_reliability_dataframe(
        test_calibrated["label"].to_numpy(),
        test_calibrated["calibrated_confidence"].to_numpy(),
        n_bins=10
    )

    save_table(before_rel, tables_dir / "reliability_before_calibration.csv")
    save_table(after_rel, tables_dir / "reliability_after_calibration.csv")
    print("Saved reliability before/after tables.")

    # Step 8: plot before/after reliability
    plot_before_after_reliability(
        before_rel,
        after_rel,
        figures_dir / "reliability_before_after.png"
    )
    print("Saved reliability_before_after.png")

    print("\nDone.")
    print("Main outputs:")
    print(f"- {calibrated_dir / 'test_calibrated.jsonl'}")
    print(f"- {tables_dir / 'calibration_split_metadata.csv'}")
    print(f"- {tables_dir / 'calibration_comparison.csv'}")
    print(f"- {tables_dir / 'reliability_before_calibration.csv'}")
    print(f"- {tables_dir / 'reliability_after_calibration.csv'}")
    print(f"- {figures_dir / 'reliability_before_after.png'}")


if __name__ == "__main__":
    main()
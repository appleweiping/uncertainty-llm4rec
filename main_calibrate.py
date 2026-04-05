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
    fit_isotonic_calibrator,
    fit_platt_calibrator,
    user_level_split,
)
from src.uncertainty.verbalized_confidence import (
    add_uncertainty_from_confidence,
    normalize_confidence_column,
)
from src.utils.paths import ensure_exp_dirs


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


def resolve_prediction_splits(
    *,
    exp_name: str,
    paths,
    input_path: str | Path | None,
    valid_path: str | Path | None,
    test_path: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if valid_path is not None and test_path is not None:
        valid_df = load_jsonl(valid_path)
        test_df = load_jsonl(test_path)
        return valid_df, test_df, "explicit_valid_test_paths"

    default_valid_path = paths.predictions_dir / "valid_raw.jsonl"
    default_test_path = paths.predictions_dir / "test_raw.jsonl"

    if default_valid_path.exists() and default_test_path.exists():
        valid_df = load_jsonl(default_valid_path)
        test_df = load_jsonl(default_test_path)
        return valid_df, test_df, "predictions_valid_test_files"

    fallback_input_path = (
        Path(input_path)
        if input_path is not None
        else paths.predictions_dir / "test_raw.jsonl"
    )
    if not fallback_input_path.exists():
        raise FileNotFoundError(
            f"[{exp_name}] No usable prediction inputs found. "
            f"Checked valid/test files under {paths.predictions_dir} and fallback file {fallback_input_path}."
        )

    raw_df = load_jsonl(fallback_input_path)
    return raw_df, pd.DataFrame(), "single_file_fallback_split"


def assert_disjoint_users(valid_df: pd.DataFrame, test_df: pd.DataFrame, user_col: str = "user_id") -> None:
    if user_col not in valid_df.columns or user_col not in test_df.columns:
        return

    overlap = set(valid_df[user_col].dropna().unique()) & set(test_df[user_col].dropna().unique())
    if overlap:
        raise ValueError(f"User leakage detected between valid and test splits: {len(overlap)} overlapping users.")


def count_overlapping_users(valid_df: pd.DataFrame, test_df: pd.DataFrame, user_col: str = "user_id") -> int:
    if user_col not in valid_df.columns or user_col not in test_df.columns:
        return 0

    overlap = set(valid_df[user_col].dropna().unique()) & set(test_df[user_col].dropna().unique())
    return int(len(overlap))


def compare_metrics(
    raw_df: pd.DataFrame,
    calibrated_df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    before_metrics = compute_calibration_metrics(
        raw_df,
        confidence_col="confidence",
        target_col="is_correct",
        n_bins=10,
    )
    after_metrics = compute_calibration_metrics(
        calibrated_df,
        confidence_col="calibrated_confidence",
        target_col="is_correct",
        n_bins=10,
    )

    rows = []
    for metric_name in before_metrics.keys():
        rows.append(
            {
                "split": split_name,
                "metric": metric_name,
                "before": before_metrics[metric_name],
                "after": after_metrics[metric_name],
            }
        )
    return pd.DataFrame(rows)


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
        help="Optional fallback path to a single raw prediction jsonl when valid/test prediction files are unavailable."
    )
    parser.add_argument(
        "--valid_path",
        type=str,
        default=None,
        help="Optional explicit path to valid raw predictions jsonl."
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
        help="Optional explicit path to test raw predictions jsonl."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs",
        help="Root directory for all experiment outputs."
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

    paths = ensure_exp_dirs(args.exp_name, args.output_root)
    raw_valid_df, raw_test_df, split_mode = resolve_prediction_splits(
        exp_name=args.exp_name,
        paths=paths,
        input_path=args.input_path,
        valid_path=args.valid_path,
        test_path=args.test_path,
    )

    if split_mode == "single_file_fallback_split":
        print(f"[{args.exp_name}] Loading fallback predictions from: {args.input_path or (paths.predictions_dir / 'test_raw.jsonl')}")
        df = prepare_prediction_dataframe(raw_valid_df)
        df = normalize_confidence_column(df, input_col="confidence", output_col="confidence")
        print(f"[{args.exp_name}] Loaded {len(df)} samples from fallback raw predictions.")

        split_result = user_level_split(
            df,
            user_col="user_id",
            valid_ratio=args.valid_ratio,
            random_state=args.random_state
        )
        valid_df = split_result.valid_df
        test_df = split_result.test_df
    else:
        valid_df = prepare_prediction_dataframe(raw_valid_df)
        valid_df = normalize_confidence_column(valid_df, input_col="confidence", output_col="confidence")
        test_df = prepare_prediction_dataframe(raw_test_df)
        test_df = normalize_confidence_column(test_df, input_col="confidence", output_col="confidence")
        print(f"[{args.exp_name}] Loaded {len(valid_df)} valid samples and {len(test_df)} test samples from prediction files.")

    overlapping_users = count_overlapping_users(valid_df, test_df, user_col="user_id")
    if split_mode == "single_file_fallback_split":
        assert_disjoint_users(valid_df, test_df, user_col="user_id")

    split_meta = build_split_metadata(valid_df, test_df)
    split_meta.update(
        {
            "split_strategy": split_mode,
            "valid_ratio": float(args.valid_ratio),
            "random_state": int(args.random_state),
            "calibration_method": args.method,
            "num_overlapping_users": int(overlapping_users),
        }
    )
    save_summary_dict(split_meta, paths.tables_dir / "calibration_split_metadata.csv")

    if args.method == "isotonic":
        calibrator = fit_isotonic_calibrator(
            valid_df=valid_df,
            confidence_col="confidence",
            target_col="is_correct",
        )
    elif args.method == "platt":
        calibrator = fit_platt_calibrator(
            valid_df=valid_df,
            confidence_col="confidence",
            target_col="is_correct",
        )
    else:
        calibrator = fit_calibrator(
            valid_df=valid_df,
            method=args.method,
            confidence_col="confidence",
            target_col="is_correct",
        )
    print(f"[{args.exp_name}] Fitted {args.method} calibrator on valid split.")

    valid_calibrated = apply_calibrator(valid_df, calibrator, input_col="confidence", output_col="calibrated_confidence")
    valid_calibrated = add_uncertainty_from_confidence(valid_calibrated, confidence_col="calibrated_confidence", output_col="uncertainty")

    test_calibrated = apply_calibrator(test_df, calibrator, input_col="confidence", output_col="calibrated_confidence")
    test_calibrated = add_uncertainty_from_confidence(test_calibrated, confidence_col="calibrated_confidence", output_col="uncertainty")

    save_jsonl(test_calibrated, paths.calibrated_dir / "test_calibrated.jsonl")
    save_jsonl(valid_calibrated, paths.calibrated_dir / "valid_calibrated.jsonl")

    comparison_df = pd.concat(
        [
            compare_metrics(valid_df, valid_calibrated, split_name="valid"),
            compare_metrics(test_df, test_calibrated, split_name="test"),
        ],
        ignore_index=True,
    )
    save_table(comparison_df, paths.tables_dir / "calibration_comparison.csv")
    save_table(comparison_df, paths.calibrated_dir / "calibration_comparison.csv")

    before_rel = get_reliability_dataframe(
        test_df["is_correct"].to_numpy(),
        test_df["confidence"].to_numpy(),
        n_bins=10
    )
    after_rel = get_reliability_dataframe(
        test_calibrated["is_correct"].to_numpy(),
        test_calibrated["calibrated_confidence"].to_numpy(),
        n_bins=10
    )

    save_table(before_rel, paths.tables_dir / "reliability_before_calibration.csv")
    save_table(after_rel, paths.tables_dir / "reliability_after_calibration.csv")

    
    plot_before_after_reliability(
        before_rel,
        after_rel,
        paths.figures_dir / "reliability_before_after_calibration.png"
    )

    print(f"[{args.exp_name}] Calibration done.")
    print(f"Calibrated files saved to: {paths.calibrated_dir}")
    print(f"Tables saved to:          {paths.tables_dir}")
    print(f"Figures saved to:         {paths.figures_dir}")


if __name__ == "__main__":
    main()

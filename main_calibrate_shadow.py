from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.shadow import compute_shadow_scores
from src.shadow.eval import compute_shadow_diagnostic_metrics, prepare_shadow_dataframe, save_table
from src.uncertainty.calibration import (
    apply_calibrator,
    build_split_metadata,
    fit_calibrator,
    user_level_split,
)
from src.utils.paths import ensure_exp_dirs


def load_jsonl(path: str | Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def save_jsonl(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def _resolve_splits(paths, valid_path: str | None, test_path: str | None, input_path: str | None):
    default_valid = paths.predictions_dir / "valid_raw.jsonl"
    default_test = paths.predictions_dir / "test_raw.jsonl"
    if valid_path and test_path:
        return load_jsonl(valid_path), load_jsonl(test_path), "explicit_valid_test_paths"
    if default_valid.exists() and default_test.exists():
        return load_jsonl(default_valid), load_jsonl(default_test), "predictions_valid_test_files"
    fallback = Path(input_path) if input_path else default_test
    if not fallback.exists():
        raise FileNotFoundError(f"No valid/test shadow predictions found under {paths.predictions_dir}")
    return load_jsonl(fallback), pd.DataFrame(), "single_file_fallback_split"


def _add_calibrated_shadow_columns(df: pd.DataFrame, *, variant: str) -> pd.DataFrame:
    out = df.copy()
    calibrated_scores = []
    calibrated_uncertainties = []
    risk_adjusted_scores = []
    for record in out.to_dict("records"):
        scores = compute_shadow_scores(
            record,
            variant=variant,
            calibrated_score=float(record["shadow_calibrated_score"]),
        )
        calibrated_scores.append(scores["shadow_score"])
        calibrated_uncertainties.append(scores["shadow_uncertainty"])
        risk_adjusted_scores.append(scores["shadow_risk_adjusted_score"])
    out["shadow_calibrated_score"] = calibrated_scores
    out["shadow_uncertainty"] = calibrated_uncertainties
    out["shadow_risk_adjusted_score"] = risk_adjusted_scores
    # Compatibility columns for existing rerank code when explicit cols are not passed.
    out["calibrated_confidence"] = out["shadow_calibrated_score"]
    out["uncertainty"] = out["shadow_uncertainty"]
    return out


def _compare(before: pd.DataFrame, after: pd.DataFrame, split_name: str) -> pd.DataFrame:
    before_metrics = compute_shadow_diagnostic_metrics(before)
    after_eval = after.copy()
    after_eval["shadow_eval_score"] = after_eval["shadow_calibrated_score"].astype(float).clip(0, 1)
    after_metrics = compute_shadow_diagnostic_metrics(after_eval)
    return pd.DataFrame(
        [
            {"split": split_name, "metric": metric, "before": before_metrics[metric], "after": after_metrics[metric]}
            for metric in before_metrics
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--shadow_variant", required=True)
    parser.add_argument("--input_path", default=None)
    parser.add_argument("--valid_path", default=None)
    parser.add_argument("--test_path", default=None)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--score_col", default="shadow_score")
    parser.add_argument("--method", choices=["isotonic", "platt"], default="isotonic")
    parser.add_argument("--valid_ratio", type=float, default=0.5)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ensure_exp_dirs(args.exp_name, args.output_root)
    raw_valid, raw_test, split_mode = _resolve_splits(paths, args.valid_path, args.test_path, args.input_path)
    if split_mode == "single_file_fallback_split":
        df = prepare_shadow_dataframe(raw_valid, score_col=args.score_col)
        split = user_level_split(df, valid_ratio=args.valid_ratio, random_state=args.random_state)
        valid_df = split.valid_df
        test_df = split.test_df
    else:
        valid_df = prepare_shadow_dataframe(raw_valid, score_col=args.score_col)
        test_df = prepare_shadow_dataframe(raw_test, score_col=args.score_col)

    valid_df["shadow_score_for_calibration"] = valid_df["shadow_eval_score"]
    test_df["shadow_score_for_calibration"] = test_df["shadow_eval_score"]
    calibrator = fit_calibrator(
        valid_df,
        method=args.method,
        confidence_col="shadow_score_for_calibration",
        target_col="label",
    )
    valid_cal = apply_calibrator(
        valid_df,
        calibrator,
        input_col="shadow_score_for_calibration",
        output_col="shadow_calibrated_score",
    )
    test_cal = apply_calibrator(
        test_df,
        calibrator,
        input_col="shadow_score_for_calibration",
        output_col="shadow_calibrated_score",
    )
    valid_cal = _add_calibrated_shadow_columns(valid_cal, variant=args.shadow_variant)
    test_cal = _add_calibrated_shadow_columns(test_cal, variant=args.shadow_variant)

    save_jsonl(valid_cal, paths.calibrated_dir / "valid_calibrated.jsonl")
    save_jsonl(test_cal, paths.calibrated_dir / "test_calibrated.jsonl")

    meta = build_split_metadata(valid_df, test_df)
    meta.update({"split_strategy": split_mode, "calibration_method": args.method, "shadow_variant": args.shadow_variant})
    save_table(pd.DataFrame([meta]), paths.tables_dir / "calibration_split_metadata.csv")
    comparison = pd.concat(
        [_compare(valid_df, valid_cal, "valid"), _compare(test_df, test_cal, "test")],
        ignore_index=True,
    )
    save_table(comparison, paths.tables_dir / "calibration_comparison.csv")
    save_table(comparison, paths.calibrated_dir / "calibration_comparison.csv")
    print(f"[{args.exp_name}] Shadow calibration done.")
    print(f"[{args.exp_name}] Calibrated files saved to: {paths.calibrated_dir}")


if __name__ == "__main__":
    main()

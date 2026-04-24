from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.eval.calibration_metrics import compute_calibration_metrics
from src.uncertainty.calibration import apply_calibrator, fit_calibrator
from src.uncertainty.evidence_features import build_evidence_feature_frame
from src.uncertainty.evidence_posterior import (
    EvidencePosteriorModel,
    apply_evidence_posterior,
    fit_evidence_posterior,
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


def high_conf_error_rate(
    df: pd.DataFrame,
    confidence_col: str,
    threshold: float = 0.8,
) -> float:
    if "is_correct" not in df.columns:
        raise ValueError("Column `is_correct` is required for high_conf_error_rate.")
    confidence = df[confidence_col].astype(float).clip(0.0, 1.0)
    high_conf = confidence >= threshold
    if int(high_conf.sum()) == 0:
        return 0.0
    wrong = df["is_correct"].astype(int) == 0
    return float((high_conf & wrong).sum() / high_conf.sum())


def usable_metric_rows(
    df: pd.DataFrame,
    confidence_col: str,
    target_col: str = "is_correct",
) -> pd.DataFrame:
    out = df.copy()
    if "parse_success" in out.columns:
        out = out[out["parse_success"].astype(bool)].copy()
    required = [confidence_col, target_col, "recommend", "label"]
    available_required = [column for column in required if column in out.columns]
    out = out.dropna(subset=available_required).copy()
    if confidence_col in out.columns:
        out[confidence_col] = pd.to_numeric(out[confidence_col], errors="coerce")
        out = out.dropna(subset=[confidence_col]).copy()
    return out.reset_index(drop=True)


def _parse_success_rate(df: pd.DataFrame) -> float:
    if "parse_success" not in df.columns or len(df) == 0:
        return float("nan")
    return float(df["parse_success"].astype(bool).mean())


def _parse_failed_count(df: pd.DataFrame) -> int:
    if "parse_success" not in df.columns:
        return 0
    return int((~df["parse_success"].astype(bool)).sum())


def metrics_row(
    *,
    split: str,
    variant: str,
    df: pd.DataFrame,
    confidence_col: str,
    model: EvidencePosteriorModel | None = None,
    status: str = "ready",
    notes: str = "",
    high_conf_threshold: float = 0.8,
) -> dict[str, Any]:
    total_rows = int(len(df))
    parse_success_rate = _parse_success_rate(df)
    parse_failed_count = _parse_failed_count(df)
    eval_df = usable_metric_rows(df, confidence_col=confidence_col)
    if eval_df.empty:
        return {
            "split": split,
            "variant": variant,
            "confidence_col": confidence_col,
            "status": "no_usable_rows",
            "feature_set": model.feature_set if model is not None else "",
            "uses_fallback": bool(model.uses_fallback) if model is not None else False,
            "fallback_reason": model.fallback_reason if model is not None else "",
            "total_rows": total_rows,
            "usable_rows": 0,
            "parse_success_rate": parse_success_rate,
            "parse_failed_count": parse_failed_count,
            "high_conf_threshold": float(high_conf_threshold),
            "high_conf_error_rate": float("nan"),
            "notes": notes,
        }
    eval_df["confidence"] = eval_df[confidence_col].astype(float).clip(0.0, 1.0)
    metrics = compute_calibration_metrics(
        eval_df,
        confidence_col="confidence",
        target_col="is_correct",
        n_bins=10,
    )
    metrics.update(
        {
            "split": split,
            "variant": variant,
            "confidence_col": confidence_col,
            "status": status,
            "feature_set": model.feature_set if model is not None else "",
            "uses_fallback": bool(model.uses_fallback) if model is not None else False,
            "fallback_reason": model.fallback_reason if model is not None else "",
            "total_rows": total_rows,
            "usable_rows": int(len(eval_df)),
            "parse_success_rate": parse_success_rate,
            "parse_failed_count": parse_failed_count,
            "high_conf_threshold": float(high_conf_threshold),
            "high_conf_error_rate": high_conf_error_rate(
                eval_df,
                confidence_col="confidence",
                threshold=high_conf_threshold,
            ),
            "notes": notes,
        }
    )
    return metrics


def calibrate_raw_confidence(valid_df: pd.DataFrame, test_df: pd.DataFrame, method: str):
    valid_fit = usable_metric_rows(valid_df, confidence_col="raw_confidence")
    if valid_fit.empty:
        raise ValueError("No usable valid rows for raw confidence calibration.")
    calibrator = fit_calibrator(
        valid_df=valid_fit,
        method=method,
        confidence_col="raw_confidence",
        target_col="is_correct",
    )
    valid_out = apply_calibrator_to_usable(valid_df, calibrator, input_col="raw_confidence", output_col="raw_calibrated_confidence")
    test_out = apply_calibrator_to_usable(test_df, calibrator, input_col="raw_confidence", output_col="raw_calibrated_confidence")
    return valid_out, test_out


def apply_calibrator_to_usable(
    df: pd.DataFrame,
    calibrator,
    input_col: str,
    output_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out[output_col] = float("nan")
    mask = pd.to_numeric(out[input_col], errors="coerce").notna()
    if "parse_success" in out.columns:
        mask = mask & out["parse_success"].astype(bool)
    if not bool(mask.any()):
        return out
    calibrated = apply_calibrator(
        out.loc[mask].copy(),
        calibrator,
        input_col=input_col,
        output_col=output_col,
    )
    out.loc[mask, output_col] = calibrated[output_col].to_numpy()
    return out


def mask_failed_repaired_rows(
    df: pd.DataFrame,
    confidence_col: str,
    uncertainty_col: str,
) -> pd.DataFrame:
    out = df.copy()
    if "parse_success" in out.columns:
        failed_mask = ~out["parse_success"].astype(bool)
        out.loc[failed_mask, [confidence_col, uncertainty_col, "evidence_uncertainty"]] = float("nan")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Evidence experiment name.")
    parser.add_argument("--output_root", type=str, default="output-repaired", help="Output root.")
    parser.add_argument("--valid_path", type=str, default=None, help="Optional valid raw jsonl.")
    parser.add_argument("--test_path", type=str, default=None, help="Optional test raw jsonl.")
    parser.add_argument(
        "--raw_calibration_method",
        type=str,
        default="isotonic",
        choices=["isotonic", "platt"],
        help="Calibration method for raw_confidence baseline.",
    )
    parser.add_argument(
        "--feature_set",
        type=str,
        default="both",
        choices=["minimal", "full", "both"],
        help="Evidence posterior feature set to fit.",
    )
    parser.add_argument(
        "--no_isotonic",
        action="store_true",
        help="Disable the second-stage isotonic calibration after logistic posterior.",
    )
    parser.add_argument(
        "--high_conf_threshold",
        type=float,
        default=0.8,
        help="Threshold used for high-confidence error rate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ensure_exp_dirs(args.exp_name, args.output_root)

    valid_path = Path(args.valid_path) if args.valid_path else paths.predictions_dir / "valid_raw.jsonl"
    test_path = Path(args.test_path) if args.test_path else paths.predictions_dir / "test_raw.jsonl"
    if not valid_path.exists():
        raise FileNotFoundError(f"Valid raw predictions not found: {valid_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test raw predictions not found: {test_path}")

    raw_valid = load_jsonl(valid_path)
    raw_test = load_jsonl(test_path)
    valid_df = build_evidence_feature_frame(raw_valid)
    test_df = build_evidence_feature_frame(raw_test)
    print(f"[{args.exp_name}] Loaded {len(valid_df)} valid and {len(test_df)} test evidence rows.")
    print(f"[{args.exp_name}] Valid parse_success={valid_df['parse_success'].mean():.4f}; test parse_success={test_df['parse_success'].mean():.4f}")

    valid_raw_calibrated, test_raw_calibrated = calibrate_raw_confidence(
        valid_df,
        test_df,
        method=args.raw_calibration_method,
    )

    feature_sets = ["minimal", "full"] if args.feature_set == "both" else [args.feature_set]
    repaired_outputs: dict[str, tuple[EvidencePosteriorModel, pd.DataFrame, pd.DataFrame]] = {}
    for feature_set in feature_sets:
        model = fit_evidence_posterior(
            valid_df,
            feature_set=feature_set,
            use_isotonic=not args.no_isotonic,
        )
        valid_repaired = apply_evidence_posterior(
            valid_df,
            model,
            output_col=f"{feature_set}_repaired_confidence",
        )
        valid_repaired[f"{feature_set}_evidence_uncertainty"] = (
            1.0 - valid_repaired[f"{feature_set}_repaired_confidence"]
        )
        valid_repaired = mask_failed_repaired_rows(
            valid_repaired,
            confidence_col=f"{feature_set}_repaired_confidence",
            uncertainty_col=f"{feature_set}_evidence_uncertainty",
        )
        test_repaired = apply_evidence_posterior(
            test_df,
            model,
            output_col=f"{feature_set}_repaired_confidence",
        )
        test_repaired[f"{feature_set}_evidence_uncertainty"] = (
            1.0 - test_repaired[f"{feature_set}_repaired_confidence"]
        )
        test_repaired = mask_failed_repaired_rows(
            test_repaired,
            confidence_col=f"{feature_set}_repaired_confidence",
            uncertainty_col=f"{feature_set}_evidence_uncertainty",
        )
        repaired_outputs[feature_set] = (model, valid_repaired, test_repaired)
        print(f"[{args.exp_name}] Fitted {feature_set} evidence posterior: status={model.status}; fallback={model.fallback_reason or 'none'}")

    rows: list[dict[str, Any]] = []
    for split, raw_df, raw_calibrated_df in (
        ("valid", valid_df, valid_raw_calibrated),
        ("test", test_df, test_raw_calibrated),
    ):
        rows.append(
            metrics_row(
                split=split,
                variant="raw_confidence_uncalibrated",
                df=raw_df,
                confidence_col="raw_confidence",
                high_conf_threshold=args.high_conf_threshold,
                notes="Direct model-reported evidence raw_confidence.",
            )
        )
        rows.append(
            metrics_row(
                split=split,
                variant="raw_confidence_calibrated",
                df=raw_calibrated_df,
                confidence_col="raw_calibrated_confidence",
                high_conf_threshold=args.high_conf_threshold,
                notes=f"Valid-set {args.raw_calibration_method} calibration over raw_confidence.",
            )
        )

    for feature_set, (model, valid_repaired, test_repaired) in repaired_outputs.items():
        for split, repaired_df in (("valid", valid_repaired), ("test", test_repaired)):
            rows.append(
                metrics_row(
                    split=split,
                    variant=f"evidence_posterior_{feature_set}",
                    df=repaired_df,
                    confidence_col=f"{feature_set}_repaired_confidence",
                    model=model,
                    status=model.status,
                    high_conf_threshold=args.high_conf_threshold,
                    notes="Logistic evidence posterior with optional isotonic calibration.",
                )
            )

    comparison_df = pd.DataFrame(rows)
    save_table(comparison_df, paths.tables_dir / "evidence_posterior_calibration_comparison.csv")

    save_jsonl(valid_raw_calibrated, paths.calibrated_dir / "raw_confidence_valid_calibrated.jsonl")
    save_jsonl(test_raw_calibrated, paths.calibrated_dir / "raw_confidence_test_calibrated.jsonl")
    for feature_set, (_, valid_repaired, test_repaired) in repaired_outputs.items():
        save_jsonl(
            valid_repaired,
            paths.calibrated_dir / f"evidence_posterior_{feature_set}_valid.jsonl",
        )
        save_jsonl(
            test_repaired,
            paths.calibrated_dir / f"evidence_posterior_{feature_set}_test.jsonl",
        )

    if "minimal" in repaired_outputs:
        _, _, minimal_test = repaired_outputs["minimal"]
        canonical_test = minimal_test.copy()
        canonical_test["repaired_confidence"] = canonical_test["minimal_repaired_confidence"]
        canonical_test["evidence_uncertainty"] = canonical_test["minimal_evidence_uncertainty"]
        save_jsonl(
            canonical_test,
            paths.calibrated_dir / "evidence_posterior_test.jsonl",
        )

    metadata = {
        "exp_name": args.exp_name,
        "output_root": args.output_root,
        "valid_path": str(valid_path),
        "test_path": str(test_path),
        "num_valid_samples": int(len(valid_df)),
        "num_test_samples": int(len(test_df)),
        "valid_usable_rows": int(len(usable_metric_rows(valid_df, confidence_col="raw_confidence"))),
        "test_usable_rows": int(len(usable_metric_rows(test_df, confidence_col="raw_confidence"))),
        "valid_parse_failed_count": _parse_failed_count(valid_df),
        "test_parse_failed_count": _parse_failed_count(test_df),
        "valid_parse_success_rate": float(valid_df["parse_success"].mean()),
        "test_parse_success_rate": float(test_df["parse_success"].mean()),
        "raw_calibration_method": args.raw_calibration_method,
        "feature_set": args.feature_set,
        "use_isotonic_after_logistic": not args.no_isotonic,
        "high_conf_threshold": float(args.high_conf_threshold),
    }
    save_table(pd.DataFrame([metadata]), paths.tables_dir / "evidence_posterior_metadata.csv")

    print(f"[{args.exp_name}] Evidence posterior calibration done.")
    print(f"Calibrated files saved to: {paths.calibrated_dir}")
    print(f"Tables saved to:          {paths.tables_dir}")


if __name__ == "__main__":
    main()

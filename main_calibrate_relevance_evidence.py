from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.eval.calibration_metrics import compute_calibration_metrics
from src.uncertainty.calibration import ConstantCalibrator, apply_calibrator, fit_calibrator
from src.utils.paths import ensure_exp_dirs


MINIMAL_FEATURES = [
    "relevance_probability",
    "abs_evidence_margin",
    "ambiguity",
    "missing_information",
]
FULL_FEATURES = [
    "relevance_probability",
    "positive_evidence",
    "negative_evidence",
    "evidence_margin",
    "abs_evidence_margin",
    "ambiguity",
    "missing_information",
    "evidence_risk",
    "confidence_margin_gap",
]


@dataclass
class RelevancePosterior:
    feature_set: str
    feature_cols: list[str]
    medians: dict[str, float]
    logistic_model: Any | None = None
    isotonic_model: Any | None = None
    fallback_model: Any | None = None
    fallback_reason: str = ""
    status: str = "ready"

    @property
    def uses_fallback(self) -> bool:
        return self.fallback_model is not None


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


def _coerce_unit(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float).clip(0.0, 1.0)


def build_relevance_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in [
        "relevance_probability",
        "positive_evidence",
        "negative_evidence",
        "ambiguity",
        "missing_information",
    ]:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = _coerce_unit(out[column])

    if "parse_success" not in out.columns:
        out["parse_success"] = out[
            [
                "relevance_probability",
                "positive_evidence",
                "negative_evidence",
                "ambiguity",
                "missing_information",
            ]
        ].notna().all(axis=1)
    else:
        out["parse_success"] = out["parse_success"].astype(bool)

    out["evidence_margin"] = out["positive_evidence"] - out["negative_evidence"]
    out["abs_evidence_margin"] = out["evidence_margin"].abs()
    out["evidence_risk"] = (
        1.0 - out["abs_evidence_margin"] + out["ambiguity"] + out["missing_information"]
    ) / 3.0
    out["evidence_risk"] = out["evidence_risk"].astype(float).clip(0.0, 1.0)
    out["confidence_margin_gap"] = out["relevance_probability"] - out["abs_evidence_margin"]

    if "label" in out.columns:
        out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int)
    out["pred_label"] = (out["relevance_probability"] >= 0.5).astype(int)
    if "recommend" not in out.columns:
        out["recommend"] = out["pred_label"].map({1: "yes", 0: "no"})
    out["is_correct"] = (out["pred_label"] == out["label"]).astype(int)
    return out


def usable_rows(df: pd.DataFrame, score_col: str, target_col: str = "label") -> pd.DataFrame:
    out = df.copy()
    if "parse_success" in out.columns:
        out = out[out["parse_success"].astype(bool)].copy()
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    return out.dropna(subset=[score_col, target_col]).reset_index(drop=True)


def feature_columns(feature_set: str) -> list[str]:
    if feature_set == "minimal":
        return MINIMAL_FEATURES.copy()
    if feature_set == "full":
        return FULL_FEATURES.copy()
    raise ValueError("feature_set must be minimal or full.")


def feature_matrix(df: pd.DataFrame, columns: list[str], medians: dict[str, float]) -> np.ndarray:
    out = build_relevance_frame(df)
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(medians[column])
    return out[columns].astype(float).to_numpy()


def fit_relevance_posterior(
    valid_df: pd.DataFrame,
    feature_set: str,
    use_isotonic: bool = True,
    min_samples: int = 20,
) -> RelevancePosterior:
    columns = feature_columns(feature_set)
    frame = usable_rows(build_relevance_frame(valid_df), "relevance_probability", target_col="label")
    frame = frame.dropna(subset=columns + ["label"]).reset_index(drop=True)
    if len(frame) < min_samples or frame["label"].nunique() < 2:
        fallback_reason = (
            f"too few usable rows: {len(frame)} < {min_samples}"
            if len(frame) < min_samples
            else "single target class"
        )
        if len(frame) == 0:
            fallback = ConstantCalibrator(0.5)
        elif frame["label"].nunique() < 2:
            fallback = ConstantCalibrator(float(frame["label"].iloc[0]))
        else:
            fallback = fit_calibrator(
                frame,
                method="isotonic",
                confidence_col="relevance_probability",
                target_col="label",
            )
        return RelevancePosterior(
            feature_set=feature_set,
            feature_cols=["relevance_probability"],
            medians={"relevance_probability": float(frame["relevance_probability"].median()) if len(frame) else 0.5},
            fallback_model=fallback,
            fallback_reason=fallback_reason,
            status="fallback",
        )

    medians = {
        column: float(pd.to_numeric(frame[column], errors="coerce").median())
        if not pd.isna(pd.to_numeric(frame[column], errors="coerce").median())
        else 0.5
        for column in columns
    }
    x = feature_matrix(frame, columns, medians)
    y = frame["label"].astype(int).to_numpy()

    try:
        logistic = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
        logistic.fit(x, y)
        prob = np.clip(logistic.predict_proba(x)[:, 1], 0.0, 1.0)
        isotonic = None
        if use_isotonic:
            isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            isotonic.fit(prob, y.astype(float))
        return RelevancePosterior(
            feature_set=feature_set,
            feature_cols=columns,
            medians=medians,
            logistic_model=logistic,
            isotonic_model=isotonic,
        )
    except Exception as exc:
        fallback = fit_calibrator(
            frame,
            method="isotonic",
            confidence_col="relevance_probability",
            target_col="label",
        )
        return RelevancePosterior(
            feature_set=feature_set,
            feature_cols=["relevance_probability"],
            medians={"relevance_probability": float(frame["relevance_probability"].median())},
            fallback_model=fallback,
            fallback_reason=f"logistic relevance posterior failed: {exc}",
            status="fallback",
        )


def predict_relevance_posterior(df: pd.DataFrame, model: RelevancePosterior) -> np.ndarray:
    frame = build_relevance_frame(df)
    if model.fallback_model is not None:
        scores = (
            pd.to_numeric(frame["relevance_probability"], errors="coerce")
            .fillna(model.medians.get("relevance_probability", 0.5))
            .astype(float)
            .clip(0.0, 1.0)
            .to_numpy()
        )
        return np.clip(model.fallback_model.predict(scores), 0.0, 1.0)
    if model.logistic_model is None:
        raise RuntimeError("RelevancePosterior has no fitted model.")
    x = feature_matrix(frame, model.feature_cols, model.medians)
    prob = np.clip(model.logistic_model.predict_proba(x)[:, 1], 0.0, 1.0)
    if model.isotonic_model is not None:
        prob = np.clip(model.isotonic_model.predict(prob), 0.0, 1.0)
    return prob


def apply_relevance_posterior(df: pd.DataFrame, model: RelevancePosterior, output_col: str) -> pd.DataFrame:
    out = build_relevance_frame(df)
    out[output_col] = predict_relevance_posterior(out, model).astype(float).clip(0.0, 1.0)
    out["relevance_uncertainty"] = 1.0 - out[output_col]
    out["relevance_posterior_feature_set"] = model.feature_set
    out["relevance_posterior_status"] = model.status
    out["relevance_posterior_fallback_reason"] = model.fallback_reason
    if "parse_success" in out.columns:
        failed = ~out["parse_success"].astype(bool)
        out.loc[failed, [output_col, "relevance_uncertainty"]] = np.nan
    return out


def apply_score_calibrator(df: pd.DataFrame, calibrator, input_col: str, output_col: str) -> pd.DataFrame:
    out = build_relevance_frame(df)
    out[output_col] = np.nan
    mask = out["parse_success"].astype(bool) & pd.to_numeric(out[input_col], errors="coerce").notna()
    if bool(mask.any()):
        calibrated = apply_calibrator(out.loc[mask].copy(), calibrator, input_col=input_col, output_col=output_col)
        out.loc[mask, output_col] = calibrated[output_col].to_numpy()
    return out


def high_conf_error_rate(df: pd.DataFrame, score_col: str, threshold: float) -> float:
    work = usable_rows(build_relevance_frame(df), score_col, target_col="label")
    if work.empty:
        return float("nan")
    high = pd.to_numeric(work[score_col], errors="coerce") >= threshold
    if int(high.sum()) == 0:
        return 0.0
    pred = (pd.to_numeric(work[score_col], errors="coerce") >= 0.5).astype(int)
    wrong = pred != work["label"].astype(int)
    return float((high & wrong).sum() / high.sum())


def metrics_row(
    *,
    split: str,
    variant: str,
    df: pd.DataFrame,
    score_col: str,
    status: str = "ready",
    model: RelevancePosterior | None = None,
    threshold: float = 0.8,
) -> dict[str, Any]:
    work = usable_rows(build_relevance_frame(df), score_col, target_col="label")
    if work.empty:
        return {
            "split": split,
            "variant": variant,
            "score_col": score_col,
            "status": "no_usable_rows",
            "feature_set": model.feature_set if model else "",
            "uses_fallback": bool(model.uses_fallback) if model else False,
            "fallback_reason": model.fallback_reason if model else "",
            "total_rows": int(len(df)),
            "usable_rows": 0,
            "parse_success_rate": float(df["parse_success"].astype(bool).mean()) if "parse_success" in df.columns else float("nan"),
            "high_conf_error_rate": float("nan"),
        }
    work["confidence"] = pd.to_numeric(work[score_col], errors="coerce").clip(0.0, 1.0)
    work["pred_label"] = (work["confidence"] >= 0.5).astype(int)
    work["recommend"] = work["pred_label"].map({1: "yes", 0: "no"})
    work["is_correct"] = (work["pred_label"] == work["label"].astype(int)).astype(int)
    metrics = compute_calibration_metrics(work, confidence_col="confidence", target_col="label")
    metrics.update(
        {
            "split": split,
            "variant": variant,
            "score_col": score_col,
            "status": status,
            "feature_set": model.feature_set if model else "",
            "uses_fallback": bool(model.uses_fallback) if model else False,
            "fallback_reason": model.fallback_reason if model else "",
            "total_rows": int(len(df)),
            "usable_rows": int(len(work)),
            "parse_success_rate": float(df["parse_success"].astype(bool).mean()) if "parse_success" in df.columns else float("nan"),
            "high_conf_threshold": float(threshold),
            "high_conf_error_rate": high_conf_error_rate(work, "confidence", threshold),
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="output-repaired")
    parser.add_argument("--valid_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--feature_set", type=str, default="both", choices=["minimal", "full", "both"])
    parser.add_argument("--raw_calibration_method", type=str, default="isotonic", choices=["isotonic", "platt"])
    parser.add_argument("--no_isotonic", action="store_true")
    parser.add_argument("--high_conf_threshold", type=float, default=0.8)
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

    valid_df = build_relevance_frame(load_jsonl(valid_path))
    test_df = build_relevance_frame(load_jsonl(test_path))
    print(
        f"[{args.exp_name}] Loaded valid={len(valid_df)} test={len(test_df)} "
        f"parse_success_valid={valid_df['parse_success'].mean():.4f} "
        f"parse_success_test={test_df['parse_success'].mean():.4f}"
    )

    valid_fit = usable_rows(valid_df, "relevance_probability", target_col="label")
    raw_calibrator = fit_calibrator(
        valid_fit,
        method=args.raw_calibration_method,
        confidence_col="relevance_probability",
        target_col="label",
    )
    valid_raw_calibrated = apply_score_calibrator(
        valid_df,
        raw_calibrator,
        input_col="relevance_probability",
        output_col="calibrated_relevance_probability",
    )
    valid_raw_calibrated["relevance_uncertainty"] = 1.0 - valid_raw_calibrated["calibrated_relevance_probability"]
    test_raw_calibrated = apply_score_calibrator(
        test_df,
        raw_calibrator,
        input_col="relevance_probability",
        output_col="calibrated_relevance_probability",
    )
    test_raw_calibrated["relevance_uncertainty"] = 1.0 - test_raw_calibrated["calibrated_relevance_probability"]

    feature_sets = ["minimal", "full"] if args.feature_set == "both" else [args.feature_set]
    models: dict[str, RelevancePosterior] = {}
    repaired_valid: dict[str, pd.DataFrame] = {}
    repaired_test: dict[str, pd.DataFrame] = {}
    for feature_set in feature_sets:
        model = fit_relevance_posterior(
            valid_df,
            feature_set=feature_set,
            use_isotonic=not args.no_isotonic,
        )
        models[feature_set] = model
        repaired_valid[feature_set] = apply_relevance_posterior(
            valid_df,
            model,
            output_col=f"{feature_set}_calibrated_relevance_probability",
        )
        repaired_test[feature_set] = apply_relevance_posterior(
            test_df,
            model,
            output_col=f"{feature_set}_calibrated_relevance_probability",
        )
        print(
            f"[{args.exp_name}] Fitted relevance posterior {feature_set}: "
            f"status={model.status}; fallback={model.fallback_reason or 'none'}"
        )

    rows: list[dict[str, Any]] = []
    for split, raw_df, raw_cal_df in [
        ("valid", valid_df, valid_raw_calibrated),
        ("test", test_df, test_raw_calibrated),
    ]:
        rows.append(
            metrics_row(
                split=split,
                variant="raw_relevance_probability",
                df=raw_df,
                score_col="relevance_probability",
                threshold=args.high_conf_threshold,
            )
        )
        rows.append(
            metrics_row(
                split=split,
                variant="calibrated_relevance_probability",
                df=raw_cal_df,
                score_col="calibrated_relevance_probability",
                threshold=args.high_conf_threshold,
            )
        )
    for feature_set, model in models.items():
        rows.append(
            metrics_row(
                split="valid",
                variant=f"evidence_posterior_relevance_{feature_set}",
                df=repaired_valid[feature_set],
                score_col=f"{feature_set}_calibrated_relevance_probability",
                model=model,
                status=model.status,
                threshold=args.high_conf_threshold,
            )
        )
        rows.append(
            metrics_row(
                split="test",
                variant=f"evidence_posterior_relevance_{feature_set}",
                df=repaired_test[feature_set],
                score_col=f"{feature_set}_calibrated_relevance_probability",
                model=model,
                status=model.status,
                threshold=args.high_conf_threshold,
            )
        )

    comparison = pd.DataFrame(rows)
    save_table(comparison, paths.tables_dir / "relevance_evidence_calibration_comparison.csv")
    save_jsonl(valid_raw_calibrated, paths.calibrated_dir / "raw_relevance_valid_calibrated.jsonl")
    save_jsonl(test_raw_calibrated, paths.calibrated_dir / "raw_relevance_test_calibrated.jsonl")
    for feature_set in feature_sets:
        save_jsonl(
            repaired_valid[feature_set],
            paths.calibrated_dir / f"relevance_evidence_posterior_{feature_set}_valid.jsonl",
        )
        save_jsonl(
            repaired_test[feature_set],
            paths.calibrated_dir / f"relevance_evidence_posterior_{feature_set}_test.jsonl",
        )

    canonical_feature = "minimal" if "minimal" in repaired_test else feature_sets[0]
    canonical = repaired_test[canonical_feature].copy()
    canonical["calibrated_relevance_probability"] = canonical[
        f"{canonical_feature}_calibrated_relevance_probability"
    ]
    canonical["relevance_uncertainty"] = 1.0 - canonical["calibrated_relevance_probability"]
    save_jsonl(canonical, paths.calibrated_dir / "relevance_evidence_posterior_test.jsonl")

    metadata = {
        "exp_name": args.exp_name,
        "output_root": args.output_root,
        "valid_path": str(valid_path),
        "test_path": str(test_path),
        "num_valid_samples": int(len(valid_df)),
        "num_test_samples": int(len(test_df)),
        "valid_parse_success_rate": float(valid_df["parse_success"].mean()),
        "test_parse_success_rate": float(test_df["parse_success"].mean()),
        "raw_calibration_method": args.raw_calibration_method,
        "feature_set": args.feature_set,
        "use_isotonic_after_logistic": not args.no_isotonic,
        "canonical_feature_set": canonical_feature,
    }
    save_table(pd.DataFrame([metadata]), paths.tables_dir / "relevance_evidence_metadata.csv")
    print(f"[{args.exp_name}] Relevance evidence calibration done.")
    print(f"Tables saved to: {paths.tables_dir}")
    print(f"Calibrated files saved to: {paths.calibrated_dir}")


if __name__ == "__main__":
    main()

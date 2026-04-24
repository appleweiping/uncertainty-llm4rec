from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.uncertainty.calibration import ConstantCalibrator, fit_calibrator
from src.uncertainty.evidence_features import (
    build_evidence_feature_frame,
    get_evidence_feature_columns,
    get_modeling_rows,
)


@dataclass
class EvidencePosteriorModel:
    feature_set: str
    feature_cols: list[str]
    feature_medians: dict[str, float]
    logistic_model: Any | None = None
    isotonic_model: Any | None = None
    fallback_model: Any | None = None
    fallback_reason: str = ""
    target_col: str = "is_correct"
    status: str = "ready"

    @property
    def uses_fallback(self) -> bool:
        return self.fallback_model is not None


def _safe_medians(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for column in feature_cols:
        value = pd.to_numeric(df[column], errors="coerce").median()
        if pd.isna(value):
            value = 0.5
        medians[column] = float(value)
    return medians


def _feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    medians: dict[str, float],
) -> np.ndarray:
    frame = build_evidence_feature_frame(df)
    for column in feature_cols:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(medians[column])
    return frame[feature_cols].astype(float).to_numpy()


def _fit_raw_confidence_fallback(
    valid_df: pd.DataFrame,
    target_col: str,
    fallback_reason: str,
) -> EvidencePosteriorModel:
    frame = build_evidence_feature_frame(valid_df)
    if target_col not in frame.columns:
        raise ValueError(f"Column `{target_col}` not found in valid_df.")

    frame = frame.dropna(subset=["raw_confidence", target_col]).reset_index(drop=True)
    unique_targets = np.unique(frame[target_col].astype(int).to_numpy())

    if len(frame) == 0:
        calibrator = ConstantCalibrator(0.5)
        fallback_reason = f"{fallback_reason}; no usable raw_confidence rows"
    elif len(unique_targets) < 2:
        calibrator = ConstantCalibrator(float(unique_targets[0]))
        fallback_reason = f"{fallback_reason}; single target class"
    else:
        calibrator = fit_calibrator(
            frame,
            method="isotonic",
            confidence_col="raw_confidence",
            target_col=target_col,
        )

    return EvidencePosteriorModel(
        feature_set="raw_confidence_fallback",
        feature_cols=["raw_confidence"],
        feature_medians={"raw_confidence": float(frame["raw_confidence"].median()) if len(frame) else 0.5},
        fallback_model=calibrator,
        fallback_reason=fallback_reason,
        target_col=target_col,
        status="fallback",
    )


def fit_evidence_posterior(
    valid_df: pd.DataFrame,
    feature_set: str = "minimal",
    target_col: str = "is_correct",
    use_isotonic: bool = True,
    min_samples: int = 20,
) -> EvidencePosteriorModel:
    feature_cols = get_evidence_feature_columns(feature_set)
    modeling_df = get_modeling_rows(valid_df, feature_set=feature_set, target_col=target_col)

    if len(modeling_df) < min_samples:
        return _fit_raw_confidence_fallback(
            valid_df,
            target_col=target_col,
            fallback_reason=f"too few usable evidence rows: {len(modeling_df)} < {min_samples}",
        )

    y = modeling_df[target_col].astype(int).to_numpy()
    unique_targets = np.unique(y)
    if len(unique_targets) < 2:
        return _fit_raw_confidence_fallback(
            valid_df,
            target_col=target_col,
            fallback_reason="single target class in usable evidence rows",
        )

    medians = _safe_medians(modeling_df, feature_cols)
    x = _feature_matrix(modeling_df, feature_cols, medians)

    try:
        logistic = LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=42,
        )
        logistic.fit(x, y)
        logistic_prob = np.clip(logistic.predict_proba(x)[:, 1], 0.0, 1.0)

        isotonic = None
        if use_isotonic:
            isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            isotonic.fit(logistic_prob, y.astype(float))

        return EvidencePosteriorModel(
            feature_set=feature_set,
            feature_cols=feature_cols,
            feature_medians=medians,
            logistic_model=logistic,
            isotonic_model=isotonic,
            target_col=target_col,
            status="ready",
        )
    except Exception as exc:
        return _fit_raw_confidence_fallback(
            valid_df,
            target_col=target_col,
            fallback_reason=f"logistic evidence posterior failed: {exc}",
        )


def predict_evidence_posterior(df: pd.DataFrame, model: EvidencePosteriorModel) -> np.ndarray:
    if model.fallback_model is not None:
        frame = build_evidence_feature_frame(df)
        raw_confidence = (
            pd.to_numeric(frame["raw_confidence"], errors="coerce")
            .fillna(model.feature_medians.get("raw_confidence", 0.5))
            .astype(float)
            .clip(0.0, 1.0)
            .to_numpy()
        )
        return np.clip(model.fallback_model.predict(raw_confidence), 0.0, 1.0)

    if model.logistic_model is None:
        raise RuntimeError("EvidencePosteriorModel has neither logistic_model nor fallback_model.")

    x = _feature_matrix(df, model.feature_cols, model.feature_medians)
    prob = np.clip(model.logistic_model.predict_proba(x)[:, 1], 0.0, 1.0)
    if model.isotonic_model is not None:
        prob = np.clip(model.isotonic_model.predict(prob), 0.0, 1.0)
    return prob


def apply_evidence_posterior(
    df: pd.DataFrame,
    model: EvidencePosteriorModel,
    output_col: str = "repaired_confidence",
) -> pd.DataFrame:
    out = build_evidence_feature_frame(df)
    out[output_col] = predict_evidence_posterior(out, model).astype(float)
    out[output_col] = out[output_col].clip(0.0, 1.0)
    out["evidence_uncertainty"] = 1.0 - out[output_col]
    out["evidence_posterior_feature_set"] = model.feature_set
    out["evidence_posterior_status"] = model.status
    out["evidence_posterior_fallback_reason"] = model.fallback_reason
    return out

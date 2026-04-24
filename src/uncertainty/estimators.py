# src/uncertainty/estimators.py

from __future__ import annotations

import pandas as pd


DEFAULT_KEY_COLS = ["user_id", "target_item_id", "candidate_item_id", "label"]


def add_verbalized_confidence(df: pd.DataFrame) -> pd.Series:
    return df["confidence"].astype(float).clip(0.0, 1.0)


def add_calibrated_confidence(df: pd.DataFrame) -> pd.Series:
    return df["calibrated_confidence"].astype(float).clip(0.0, 1.0)


def add_raw_evidence_confidence(df: pd.DataFrame) -> pd.Series:
    return df["raw_confidence"].astype(float).clip(0.0, 1.0)


def add_raw_evidence_calibrated_confidence(df: pd.DataFrame) -> pd.Series:
    return df["raw_calibrated_confidence"].astype(float).clip(0.0, 1.0)


def add_evidence_posterior_confidence(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].astype(float).clip(0.0, 1.0)


def add_consistency_confidence(df: pd.DataFrame) -> pd.Series:
    return df["consistency_confidence"].astype(float).clip(0.0, 1.0)


def _normalize_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Fusion alpha must be in [0, 1], got {alpha}.")
    return alpha


def fuse_confidence(
    calibrated_confidence: pd.Series,
    consistency_confidence: pd.Series,
    alpha: float,
) -> pd.Series:
    alpha = _normalize_alpha(alpha)
    calibrated = calibrated_confidence.astype(float).clip(0.0, 1.0)
    consistency = consistency_confidence.astype(float).clip(0.0, 1.0)
    return alpha * calibrated + (1.0 - alpha) * consistency


def fuse_uncertainty(
    calibrated_uncertainty: pd.Series,
    consistency_uncertainty: pd.Series,
    alpha: float,
) -> pd.Series:
    alpha = _normalize_alpha(alpha)
    calibrated = calibrated_uncertainty.astype(float).clip(0.0, 1.0)
    consistency = consistency_uncertainty.astype(float).clip(0.0, 1.0)
    return alpha * calibrated + (1.0 - alpha) * consistency


def merge_consistency_outputs(
    base_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    key_cols: list[str] | None = None,
) -> pd.DataFrame:
    key_cols = key_cols or DEFAULT_KEY_COLS
    available_key_cols = [col for col in key_cols if col in base_df.columns and col in consistency_df.columns]
    if not available_key_cols:
        raise ValueError("No shared key columns available to merge consistency outputs.")

    consistency_cols = available_key_cols + [
        col
        for col in [
            "num_consistency_samples",
            "yes_count",
            "no_count",
            "unknown_count",
            "yes_ratio",
            "no_ratio",
            "unknown_ratio",
            "majority_vote",
            "majority_ratio",
            "vote_entropy",
            "vote_variance",
            "mean_confidence",
            "confidence_variance",
            "consistency_confidence",
            "consistency_uncertainty",
        ]
        if col in consistency_df.columns
    ]

    deduped = consistency_df[consistency_cols].drop_duplicates(subset=available_key_cols)
    return base_df.merge(deduped, on=available_key_cols, how="left")


def add_fused_confidence(df: pd.DataFrame, alpha: float = 0.5) -> pd.Series:
    return fuse_confidence(
        df["calibrated_confidence"],
        df["consistency_confidence"],
        alpha=alpha,
    )


def add_fused_uncertainty(df: pd.DataFrame, alpha: float = 0.5) -> pd.Series:
    return fuse_uncertainty(
        df["uncertainty"],
        df["consistency_uncertainty"],
        alpha=alpha,
    )


def ensure_estimator_columns(df: pd.DataFrame, fused_alpha: float = 0.5) -> pd.DataFrame:
    out = df.copy()

    if "confidence" in out.columns:
        out["verbalized_confidence"] = add_verbalized_confidence(out)
        out["verbalized_uncertainty"] = 1.0 - out["verbalized_confidence"]

    if "raw_confidence" in out.columns:
        out["evidence_raw_confidence"] = add_raw_evidence_confidence(out)
        out["evidence_raw_uncertainty"] = 1.0 - out["evidence_raw_confidence"]

    if "raw_calibrated_confidence" in out.columns:
        out["evidence_raw_calibrated_confidence"] = add_raw_evidence_calibrated_confidence(out)
        out["evidence_raw_calibrated_uncertainty"] = 1.0 - out["evidence_raw_calibrated_confidence"]

    if "minimal_repaired_confidence" in out.columns:
        out["evidence_posterior_minimal_confidence"] = add_evidence_posterior_confidence(
            out,
            "minimal_repaired_confidence",
        )
        if "minimal_evidence_uncertainty" in out.columns:
            out["evidence_posterior_minimal_uncertainty"] = (
                out["minimal_evidence_uncertainty"].astype(float).clip(0.0, 1.0)
            )
        else:
            out["evidence_posterior_minimal_uncertainty"] = 1.0 - out["evidence_posterior_minimal_confidence"]

    if "full_repaired_confidence" in out.columns:
        out["evidence_posterior_full_confidence"] = add_evidence_posterior_confidence(
            out,
            "full_repaired_confidence",
        )
        if "full_evidence_uncertainty" in out.columns:
            out["evidence_posterior_full_uncertainty"] = (
                out["full_evidence_uncertainty"].astype(float).clip(0.0, 1.0)
            )
        else:
            out["evidence_posterior_full_uncertainty"] = 1.0 - out["evidence_posterior_full_confidence"]

    if "calibrated_confidence" in out.columns:
        out["verbalized_calibrated_confidence"] = add_calibrated_confidence(out)
        if "uncertainty" in out.columns:
            out["verbalized_calibrated_uncertainty"] = out["uncertainty"].astype(float).clip(0.0, 1.0)
        else:
            out["verbalized_calibrated_uncertainty"] = 1.0 - out["verbalized_calibrated_confidence"]

    if "consistency_confidence" in out.columns:
        out["consistency_confidence"] = add_consistency_confidence(out)
        if "consistency_uncertainty" in out.columns:
            out["consistency_uncertainty"] = out["consistency_uncertainty"].astype(float).clip(0.0, 1.0)

    if {
        "calibrated_confidence",
        "uncertainty",
        "consistency_confidence",
        "consistency_uncertainty",
    }.issubset(out.columns):
        out["fused_confidence"] = add_fused_confidence(out, alpha=fused_alpha)
        out["fused_uncertainty"] = add_fused_uncertainty(out, alpha=fused_alpha)
        out["fused_alpha"] = fused_alpha

    return out


def get_available_estimators(df: pd.DataFrame, fused_alpha: float = 0.5) -> dict[str, dict[str, str | float]]:
    estimators: dict[str, dict[str, str | float]] = {}

    if {"verbalized_confidence", "verbalized_uncertainty"}.issubset(df.columns):
        estimators["verbalized_raw"] = {
            "confidence_col": "verbalized_confidence",
            "uncertainty_col": "verbalized_uncertainty",
        }

    if {"evidence_raw_confidence", "evidence_raw_uncertainty"}.issubset(df.columns):
        estimators["evidence_raw"] = {
            "confidence_col": "evidence_raw_confidence",
            "uncertainty_col": "evidence_raw_uncertainty",
        }

    if {"evidence_raw_calibrated_confidence", "evidence_raw_calibrated_uncertainty"}.issubset(df.columns):
        estimators["evidence_raw_calibrated"] = {
            "confidence_col": "evidence_raw_calibrated_confidence",
            "uncertainty_col": "evidence_raw_calibrated_uncertainty",
        }

    if {"evidence_posterior_minimal_confidence", "evidence_posterior_minimal_uncertainty"}.issubset(df.columns):
        estimators["evidence_posterior_minimal"] = {
            "confidence_col": "evidence_posterior_minimal_confidence",
            "uncertainty_col": "evidence_posterior_minimal_uncertainty",
        }

    if {"evidence_posterior_full_confidence", "evidence_posterior_full_uncertainty"}.issubset(df.columns):
        estimators["evidence_posterior_full"] = {
            "confidence_col": "evidence_posterior_full_confidence",
            "uncertainty_col": "evidence_posterior_full_uncertainty",
        }

    if {"verbalized_calibrated_confidence", "verbalized_calibrated_uncertainty"}.issubset(df.columns):
        estimators["verbalized_calibrated"] = {
            "confidence_col": "verbalized_calibrated_confidence",
            "uncertainty_col": "verbalized_calibrated_uncertainty",
        }

    if {"consistency_confidence", "consistency_uncertainty"}.issubset(df.columns):
        estimators["consistency"] = {
            "confidence_col": "consistency_confidence",
            "uncertainty_col": "consistency_uncertainty",
        }

    if {"fused_confidence", "fused_uncertainty"}.issubset(df.columns):
        estimators["fused"] = {
            "confidence_col": "fused_confidence",
            "uncertainty_col": "fused_uncertainty",
            "fusion_alpha": _normalize_alpha(fused_alpha),
        }

    return estimators

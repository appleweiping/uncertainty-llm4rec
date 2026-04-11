# src/uncertainty/estimators.py

from __future__ import annotations

import pandas as pd


DEFAULT_KEY_COLS = ["user_id", "target_item_id", "candidate_item_id", "label"]


def add_verbalized_confidence(df: pd.DataFrame) -> pd.Series:
    return df["confidence"].astype(float).clip(0.0, 1.0)


def add_calibrated_confidence(df: pd.DataFrame) -> pd.Series:
    return df["calibrated_confidence"].astype(float).clip(0.0, 1.0)


def add_consistency_confidence(df: pd.DataFrame) -> pd.Series:
    return df["consistency_confidence"].astype(float).clip(0.0, 1.0)


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


def add_fused_confidence(df: pd.DataFrame) -> pd.Series:
    return (
        0.5 * df["calibrated_confidence"].astype(float).clip(0.0, 1.0)
        + 0.5 * df["consistency_confidence"].astype(float).clip(0.0, 1.0)
    )


def add_fused_uncertainty(df: pd.DataFrame) -> pd.Series:
    return (
        0.5 * df["uncertainty"].astype(float).clip(0.0, 1.0)
        + 0.5 * df["consistency_uncertainty"].astype(float).clip(0.0, 1.0)
    )


def ensure_estimator_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "confidence" in out.columns:
        out["verbalized_confidence"] = add_verbalized_confidence(out)
        out["verbalized_uncertainty"] = 1.0 - out["verbalized_confidence"]

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
        out["fused_confidence"] = add_fused_confidence(out)
        out["fused_uncertainty"] = add_fused_uncertainty(out)

    return out


def get_available_estimators(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    estimators: dict[str, dict[str, str]] = {}

    if {"verbalized_confidence", "verbalized_uncertainty"}.issubset(df.columns):
        estimators["verbalized_raw"] = {
            "confidence_col": "verbalized_confidence",
            "uncertainty_col": "verbalized_uncertainty",
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
        }

    return estimators

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.eval.calibration_metrics import normalize_recommend


EVIDENCE_BASE_COLUMNS = (
    "raw_confidence",
    "positive_evidence",
    "negative_evidence",
    "ambiguity",
    "missing_information",
)

MINIMAL_EVIDENCE_FEATURES = (
    "raw_confidence",
    "abs_evidence_margin",
    "ambiguity",
    "missing_information",
)

FULL_EVIDENCE_FEATURES = (
    "raw_confidence",
    "positive_evidence",
    "negative_evidence",
    "evidence_margin",
    "abs_evidence_margin",
    "ambiguity",
    "missing_information",
    "confidence_margin_gap",
)


def get_evidence_feature_columns(feature_set: str = "minimal") -> list[str]:
    name = str(feature_set).strip().lower()
    if name == "minimal":
        return list(MINIMAL_EVIDENCE_FEATURES)
    if name == "full":
        return list(FULL_EVIDENCE_FEATURES)
    raise ValueError("feature_set must be either 'minimal' or 'full'.")


def _coerce_unit_interval(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float).clip(0.0, 1.0)


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out


def build_evidence_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_columns(df, EVIDENCE_BASE_COLUMNS)

    for column in EVIDENCE_BASE_COLUMNS:
        out[column] = _coerce_unit_interval(out[column])

    if "parse_success" not in out.columns:
        out["parse_success"] = out[list(EVIDENCE_BASE_COLUMNS)].notna().all(axis=1)
    else:
        out["parse_success"] = out["parse_success"].astype(bool)

    out["evidence_margin"] = out["positive_evidence"] - out["negative_evidence"]
    out["abs_evidence_margin"] = out["evidence_margin"].abs()
    out["confidence_margin_gap"] = out["raw_confidence"] - out["abs_evidence_margin"]

    if "pred_label" not in out.columns and "recommend" in out.columns:
        out["pred_label"] = out["recommend"].apply(normalize_recommend).astype(int)

    if "label" in out.columns:
        out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int)
        if "pred_label" in out.columns:
            out["is_correct"] = (out["pred_label"] == out["label"]).astype(int)

    return out


def get_modeling_rows(
    df: pd.DataFrame,
    feature_set: str = "minimal",
    target_col: str = "is_correct",
) -> pd.DataFrame:
    out = build_evidence_feature_frame(df)
    feature_cols = get_evidence_feature_columns(feature_set)
    required_cols = feature_cols + [target_col]
    if "parse_success" in out.columns:
        out = out[out["parse_success"].astype(bool)].copy()
    return out.dropna(subset=required_cols).reset_index(drop=True)

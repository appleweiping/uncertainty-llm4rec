# src/uncertainty/estimators.py

from __future__ import annotations

import pandas as pd


def add_verbalized_confidence(df: pd.DataFrame):
    return df["confidence"].astype(float)


def add_calibrated_confidence(df: pd.DataFrame):
    return df["calibrated_confidence"].astype(float)


def add_consistency_confidence(df: pd.DataFrame):
    return df["consistency_confidence"].astype(float)


def add_fused_confidence(df: pd.DataFrame):
    """
    simple fusion
    """
    return (
        0.5 * df["calibrated_confidence"]
        + 0.5 * df["consistency_confidence"]
    )
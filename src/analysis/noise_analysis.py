# src/analysis/noise_analysis.py

from __future__ import annotations

import pandas as pd


def summarize_noise_effect(df: pd.DataFrame):
    summary = {}

    for col in df.columns:
        if col.endswith("_drop"):
            summary[col] = df[col].mean()

    return summary
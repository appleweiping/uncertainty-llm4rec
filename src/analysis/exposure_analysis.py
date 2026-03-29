# src/analysis/exposure_analysis.py

from __future__ import annotations

import pandas as pd

from src.analysis.confidence_correctness import prepare_prediction_dataframe


def compute_high_confidence_exposure(
    df: pd.DataFrame,
    high_conf_threshold: float = 0.8
) -> pd.DataFrame:
    df = prepare_prediction_dataframe(df)

    all_dist = (
        df["target_popularity_group"]
        .value_counts(normalize=True)
        .rename("overall_fraction")
        .reset_index()
        .rename(columns={"index": "target_popularity_group"})
    )

    high_conf_df = df[df["confidence"] >= high_conf_threshold]
    high_dist = (
        high_conf_df["target_popularity_group"]
        .value_counts(normalize=True)
        .rename("high_conf_fraction")
        .reset_index()
        .rename(columns={"index": "target_popularity_group"})
    )

    merged = all_dist.merge(high_dist, on="target_popularity_group", how="outer").fillna(0.0)
    merged["exposure_shift"] = merged["high_conf_fraction"] - merged["overall_fraction"]

    order = {"head": 0, "mid": 1, "tail": 2, "unknown": 3}
    merged["sort_key"] = merged["target_popularity_group"].map(order).fillna(999)
    merged = merged.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)

    return merged
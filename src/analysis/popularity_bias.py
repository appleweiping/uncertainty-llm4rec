# src/analysis/popularity_bias.py

from __future__ import annotations

import pandas as pd

from src.analysis.confidence_correctness import prepare_prediction_dataframe


def compute_popularity_group_stats(
    df: pd.DataFrame,
    high_conf_threshold: float = 0.8
) -> pd.DataFrame:
    df = prepare_prediction_dataframe(df)

    rows = []
    for group_name, group_df in df.groupby("target_popularity_group"):
        rows.append(
            {
                "target_popularity_group": group_name,
                "num_samples": int(len(group_df)),
                "avg_confidence": float(group_df["confidence"].mean()),
                "median_confidence": float(group_df["confidence"].median()),
                "avg_accuracy": float(group_df["is_correct"].mean()),
                "high_conf_fraction": float((group_df["confidence"] >= high_conf_threshold).mean()),
                "wrong_high_conf_fraction": float(
                    ((group_df["is_correct"] == 0) & (group_df["confidence"] >= high_conf_threshold)).mean()
                ),
            }
        )

    result = pd.DataFrame(rows)

    order = {"head": 0, "mid": 1, "tail": 2, "unknown": 3}
    if not result.empty:
        result["sort_key"] = result["target_popularity_group"].map(order).fillna(999)
        result = result.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)

    return result
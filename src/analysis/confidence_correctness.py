# src/analysis/confidence_correctness.py

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.eval.calibration_metrics import ensure_binary_columns


def prepare_prediction_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_binary_columns(df)

    if "target_popularity_group" not in out.columns:
        out["target_popularity_group"] = "unknown"

    out["target_popularity_group"] = (
        out["target_popularity_group"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
    )

    return out


def compute_confidence_correctness_summary(
    df: pd.DataFrame,
    high_conf_threshold: float = 0.8
) -> Dict[str, float]:
    df = prepare_prediction_dataframe(df)

    correct_df = df[df["is_correct"] == 1]
    wrong_df = df[df["is_correct"] == 0]
    high_conf_df = df[df["confidence"] >= high_conf_threshold]

    summary = {
        "num_samples": int(len(df)),
        "overall_accuracy": float(df["is_correct"].mean()),
        "overall_avg_confidence": float(df["confidence"].mean()),
        "correct_avg_confidence": float(correct_df["confidence"].mean()) if len(correct_df) else float("nan"),
        "wrong_avg_confidence": float(wrong_df["confidence"].mean()) if len(wrong_df) else float("nan"),
        "high_conf_fraction": float((df["confidence"] >= high_conf_threshold).mean()),
        "high_conf_accuracy": float(high_conf_df["is_correct"].mean()) if len(high_conf_df) else float("nan"),
        "wrong_high_conf_fraction": float(((df["is_correct"] == 0) & (df["confidence"] >= high_conf_threshold)).mean()),
    }
    return summary


def compute_confidence_bins_accuracy(
    df: pd.DataFrame,
    n_bins: int = 10
) -> pd.DataFrame:
    df = prepare_prediction_dataframe(df)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (df["confidence"] >= lower) & (df["confidence"] <= upper)
        else:
            mask = (df["confidence"] >= lower) & (df["confidence"] < upper)

        subset = df[mask]

        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "bin_center": float((lower + upper) / 2.0),
                "count": int(len(subset)),
                "avg_confidence": float(subset["confidence"].mean()) if len(subset) else float("nan"),
                "accuracy": float(subset["is_correct"].mean()) if len(subset) else float("nan"),
            }
        )

    return pd.DataFrame(rows)
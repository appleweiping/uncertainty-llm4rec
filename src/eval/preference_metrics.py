from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_pairwise_eval_frame(predictions_df: pd.DataFrame) -> pd.DataFrame:
    df = predictions_df.copy()

    if "preferred_item_true" not in df.columns:
        raise ValueError("Pairwise predictions must contain `preferred_item_true`.")
    if "preferred_item_pred" not in df.columns:
        raise ValueError("Pairwise predictions must contain `preferred_item_pred`.")

    df["preferred_item_true"] = df["preferred_item_true"].astype(str)
    df["preferred_item_pred"] = df["preferred_item_pred"].astype(str)
    df["confidence"] = pd.to_numeric(df.get("confidence"), errors="coerce").clip(0.0, 1.0)
    df["latency"] = pd.to_numeric(df.get("latency"), errors="coerce")
    df["parse_success"] = df.get("parse_success", False).fillna(False).astype(bool)
    df["ambiguous_preference"] = df.get("ambiguous_preference", False).fillna(False).astype(bool)
    df["is_correct"] = (df["preferred_item_pred"] == df["preferred_item_true"]).astype(int)

    return df


def compute_pairwise_metrics(pairwise_eval_df: pd.DataFrame) -> dict[str, float]:
    if pairwise_eval_df.empty:
        return {
            "sample_count": 0,
            "pairwise_accuracy": float("nan"),
            "parse_success_rate": float("nan"),
            "avg_confidence": float("nan"),
            "avg_latency": float("nan"),
            "ambiguous_preference_rate": float("nan"),
            "preference_consistency": float("nan"),
            "strict_preference_consistency": float("nan"),
        }

    df = pairwise_eval_df.copy()

    event_alignment = []
    strict_alignment = []
    if "source_event_id" in df.columns:
        for _, event_df in df.groupby("source_event_id"):
            correctness = event_df["is_correct"].astype(float)
            event_alignment.append(float(correctness.mean()))
            strict_alignment.append(float((correctness == 1.0).all()))

    return {
        "sample_count": int(len(df)),
        "pairwise_accuracy": float(df["is_correct"].mean()),
        "parse_success_rate": float(df["parse_success"].mean()),
        "avg_confidence": float(df["confidence"].mean()),
        "avg_latency": float(df["latency"].mean()),
        "ambiguous_preference_rate": float(df["ambiguous_preference"].mean()),
        "preference_consistency": float(np.mean(event_alignment)) if event_alignment else float("nan"),
        "strict_preference_consistency": float(np.mean(strict_alignment)) if strict_alignment else float("nan"),
    }


def compute_preference_confidence_bins(
    pairwise_eval_df: pd.DataFrame,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    df = pairwise_eval_df.copy()
    if df.empty:
        return pd.DataFrame(
            columns=["bin_lower", "bin_upper", "bin_center", "count", "avg_confidence", "accuracy"]
        )

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []

    for idx in range(n_bins):
        lower = float(bin_edges[idx])
        upper = float(bin_edges[idx + 1])

        if idx == n_bins - 1:
            mask = (df["confidence"] >= lower) & (df["confidence"] <= upper)
        else:
            mask = (df["confidence"] >= lower) & (df["confidence"] < upper)

        subset = df[mask]
        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "bin_center": float((lower + upper) / 2.0),
                "count": int(len(subset)),
                "avg_confidence": float(subset["confidence"].mean()) if len(subset) else float("nan"),
                "accuracy": float(subset["is_correct"].mean()) if len(subset) else float("nan"),
            }
        )

    return pd.DataFrame(rows)

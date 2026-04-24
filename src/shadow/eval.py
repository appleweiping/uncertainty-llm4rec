from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.eval.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    get_reliability_dataframe,
    roc_auc_score_manual,
)


def prepare_shadow_dataframe(
    df: pd.DataFrame,
    *,
    score_col: str = "shadow_score",
    threshold: float = 0.5,
) -> pd.DataFrame:
    out = df.copy()
    if "label" not in out.columns:
        raise ValueError("Shadow evaluation requires a `label` column.")
    if score_col not in out.columns:
        fallback_cols = ["shadow_primary_score", "confidence"]
        for fallback in fallback_cols:
            if fallback in out.columns:
                score_col = fallback
                break
        else:
            raise ValueError(f"Shadow score column not found: {score_col}")

    out["label"] = out["label"].astype(int)
    out["shadow_eval_score"] = out[score_col].astype(float).clip(0.0, 1.0)
    out["pred_label"] = (out["shadow_eval_score"] >= float(threshold)).astype(int)
    out["is_correct"] = (out["pred_label"] == out["label"]).astype(int)
    out["confidence"] = out["shadow_eval_score"]
    out["recommend"] = out["pred_label"].map({1: "yes", 0: "no"})
    if "target_popularity_group" not in out.columns:
        out["target_popularity_group"] = "unknown"
    return out


def compute_shadow_diagnostic_metrics(
    df: pd.DataFrame,
    *,
    score_col: str = "shadow_eval_score",
    target_col: str = "label",
    n_bins: int = 10,
) -> dict[str, Any]:
    y_true = df[target_col].astype(int).to_numpy()
    y_score = df[score_col].astype(float).clip(0.0, 1.0).to_numpy()
    ece, mce, _ = expected_calibration_error(y_true, y_score, n_bins=n_bins)
    return {
        "num_samples": int(len(df)),
        "accuracy": float(df["is_correct"].mean()),
        "avg_score": float(y_score.mean()),
        "avg_label": float(y_true.mean()),
        "brier_score": brier_score(y_true, y_score),
        "ece": ece,
        "mce": mce,
        "auroc": roc_auc_score_manual(y_true, y_score),
    }


def compute_shadow_score_summary(df: pd.DataFrame) -> dict[str, Any]:
    correct_df = df[df["label"] == 1]
    negative_df = df[df["label"] == 0]
    return {
        "num_samples": int(len(df)),
        "positive_avg_score": float(correct_df["shadow_eval_score"].mean()) if len(correct_df) else float("nan"),
        "negative_avg_score": float(negative_df["shadow_eval_score"].mean()) if len(negative_df) else float("nan"),
        "score_std": float(df["shadow_eval_score"].std()) if len(df) else float("nan"),
        "score_min": float(df["shadow_eval_score"].min()) if len(df) else float("nan"),
        "score_max": float(df["shadow_eval_score"].max()) if len(df) else float("nan"),
        "parse_success_rate": float(df["parse_success"].mean()) if "parse_success" in df.columns else float("nan"),
    }


def shadow_reliability_dataframe(
    df: pd.DataFrame,
    *,
    score_col: str = "shadow_eval_score",
    target_col: str = "label",
    n_bins: int = 10,
) -> pd.DataFrame:
    return get_reliability_dataframe(
        df[target_col].astype(int).to_numpy(),
        df[score_col].astype(float).clip(0.0, 1.0).to_numpy(),
        n_bins=n_bins,
    )


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

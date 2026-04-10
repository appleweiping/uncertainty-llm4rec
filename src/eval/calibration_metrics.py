# src/eval/calibration_metrics.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class BinStats:
    bin_lower: float
    bin_upper: float
    bin_center: float
    count: int
    avg_confidence: float
    accuracy: float


def normalize_recommend(value) -> int:
    """
    Convert recommend output to binary prediction:
    yes/true/1 -> 1
    no/false/0 -> 0
    """
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return int(value >= 1)

    if value is None:
        return 0

    text = str(value).strip().lower()
    if text in {"yes", "true", "1", "recommend", "positive"}:
        return 1
    if text in {"no", "false", "0", "not_recommend", "negative"}:
        return 0

    # fallback: unknown value treated as negative
    return 0


def ensure_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "pred_label" not in out.columns:
        out["pred_label"] = out["recommend"].apply(normalize_recommend)

    if "label" not in out.columns:
        raise ValueError("Input dataframe must contain `label` column.")

    out["label"] = out["label"].astype(int)
    out["confidence"] = out["confidence"].astype(float).clip(0.0, 1.0)
    out["is_correct"] = (out["pred_label"] == out["label"]).astype(int)
    return out


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Manual AUROC implementation to avoid hard dependency on sklearn.
    Computes probability that a random positive has larger score than a random negative.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")

    wins = 0.0
    total = 0.0
    for ps in pos_scores:
        wins += np.sum(ps > neg_scores)
        wins += 0.5 * np.sum(ps == neg_scores)
        total += len(neg_scores)

    return float(wins / total)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float, List[BinStats]]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_stats: List[BinStats] = []

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)

        count = int(mask.sum())

        if count == 0:
            bin_stats.append(
                BinStats(
                    bin_lower=float(lower),
                    bin_upper=float(upper),
                    bin_center=float((lower + upper) / 2.0),
                    count=0,
                    avg_confidence=float("nan"),
                    accuracy=float("nan"),
                )
            )
            continue

        avg_conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        gap = abs(acc - avg_conf)

        ece += (count / len(y_true)) * gap
        mce = max(mce, gap)

        bin_stats.append(
            BinStats(
                bin_lower=float(lower),
                bin_upper=float(upper),
                bin_center=float((lower + upper) / 2.0),
                count=count,
                avg_confidence=avg_conf,
                accuracy=acc,
            )
        )

    return float(ece), float(mce), bin_stats


def get_reliability_dataframe(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> pd.DataFrame:
    _, _, bin_stats = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    return pd.DataFrame([vars(x) for x in bin_stats])


def compute_calibration_metrics(
    df: pd.DataFrame,
    confidence_col: str = "confidence",
    target_col: str = "is_correct",
    n_bins: int = 10
) -> Dict[str, float]:
    df = ensure_binary_columns(df)

    if target_col not in df.columns:
        raise ValueError(f"Column `{target_col}` not found in dataframe.")

    y_true = df[target_col].astype(int).to_numpy()
    y_prob = df[confidence_col].astype(float).clip(0.0, 1.0).to_numpy()

    ece, mce, _ = expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    metrics = {
        "num_samples": int(len(df)),
        "accuracy": float(np.mean(df["is_correct"])),
        "avg_confidence": float(np.mean(y_prob)),
        "avg_correctness": float(np.mean(df["is_correct"])),
        "brier_score": brier_score(y_true, y_prob),
        "ece": ece,
        "mce": mce,
        "auroc": roc_auc_score_manual(y_true, y_prob),
    }
    return metrics

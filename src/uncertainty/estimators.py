from __future__ import annotations

import math
from typing import Any

import pandas as pd


DEFAULT_KEY_COLS = ["user_id", "target_item_id", "candidate_item_id", "label"]


def _clip01(series: pd.Series) -> pd.Series:
    return series.astype(float).clip(0.0, 1.0)


def add_verbalized_confidence(df: pd.DataFrame) -> pd.Series:
    return _clip01(df["confidence"])


def add_calibrated_confidence(df: pd.DataFrame) -> pd.Series:
    return _clip01(df["calibrated_confidence"])


def add_consistency_confidence(df: pd.DataFrame) -> pd.Series:
    return _clip01(df["consistency_confidence"])


def _normalize_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Fusion alpha must be in [0, 1], got {alpha}.")
    return alpha


def fuse_confidence(
    calibrated_confidence: pd.Series,
    consistency_confidence: pd.Series,
    alpha: float,
) -> pd.Series:
    alpha = _normalize_alpha(alpha)
    calibrated = _clip01(calibrated_confidence)
    consistency = _clip01(consistency_confidence)
    return alpha * calibrated + (1.0 - alpha) * consistency


def fuse_uncertainty(
    calibrated_uncertainty: pd.Series,
    consistency_uncertainty: pd.Series,
    alpha: float,
) -> pd.Series:
    alpha = _normalize_alpha(alpha)
    calibrated = _clip01(calibrated_uncertainty)
    consistency = _clip01(consistency_uncertainty)
    return alpha * calibrated + (1.0 - alpha) * consistency


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


def add_fused_confidence(df: pd.DataFrame, alpha: float = 0.5) -> pd.Series:
    return fuse_confidence(
        df["calibrated_confidence"],
        df["consistency_confidence"],
        alpha=alpha,
    )


def add_fused_uncertainty(df: pd.DataFrame, alpha: float = 0.5) -> pd.Series:
    return fuse_uncertainty(
        df["uncertainty"],
        df["consistency_uncertainty"],
        alpha=alpha,
    )


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps)
    if total <= 0:
        return [0.0 for _ in scores]
    return [value / total for value in exps]


def _normalized_entropy(probabilities: list[float]) -> float:
    if not probabilities:
        return float("nan")
    valid = [prob for prob in probabilities if prob > 0]
    if not valid:
        return 0.0
    entropy = -sum(prob * math.log(prob + 1e-12) for prob in valid)
    normalizer = math.log(len(probabilities) + 1e-12)
    if normalizer <= 0:
        return 0.0
    return float(entropy / normalizer)


def _normalize_candidate_scores(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    out: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        item_id = str(
            row.get("item_id")
            or row.get("candidate_item_id")
            or row.get("id")
            or ""
        ).strip()
        if not item_id:
            continue
        out.append(
            {
                "item_id": item_id,
                "score": _safe_float(row.get("score")),
                "reason": str(row.get("reason", "")).strip(),
            }
        )
    return out


def build_ranking_proxy_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for record in raw_df.to_dict(orient="records"):
        candidate_scores = _normalize_candidate_scores(record.get("candidate_scores"))
        if not candidate_scores:
            continue

        sorted_scores = sorted(candidate_scores, key=lambda row: row["score"], reverse=True)
        scores = [row["score"] for row in sorted_scores]
        probabilities = _softmax(scores)

        top1 = sorted_scores[0]
        top2_score = sorted_scores[1]["score"] if len(sorted_scores) >= 2 else float("nan")
        score_margin = (top1["score"] - top2_score) if len(sorted_scores) >= 2 else float("nan")
        entropy = _normalized_entropy(probabilities)
        top1_probability = probabilities[0] if probabilities else float("nan")

        target_item_id = str(record.get("target_item_id", "")).strip()
        selected_item_id = str(record.get("selected_item_id") or top1["item_id"]).strip()

        rows.append(
            {
                "task_type": "candidate_ranking",
                "user_id": str(record.get("user_id", "")).strip(),
                "target_item_id": target_item_id,
                "selected_item_id": selected_item_id,
                "candidate_count": int(record.get("candidate_count", len(sorted_scores) or 0)),
                "top1_score": _safe_float(top1["score"]),
                "top2_score": _safe_float(top2_score),
                "score_margin": _safe_float(score_margin),
                "score_entropy": _safe_float(entropy),
                "score_sharpness": _safe_float(1.0 - entropy) if not pd.isna(entropy) else float("nan"),
                "proxy_confidence": _safe_float(top1_probability),
                "label": int(selected_item_id == target_item_id and selected_item_id != ""),
                "recommend": "yes",
                "pred_label": 1,
                "target_popularity_group": str(record.get("target_popularity_group", "unknown")).strip().lower() or "unknown",
            }
        )

    return pd.DataFrame(rows)


def build_ranking_candidate_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for record in raw_df.to_dict(orient="records"):
        user_id = str(record.get("user_id", "")).strip()
        target_item_id = str(record.get("target_item_id", "")).strip()
        popularity_group = str(record.get("target_popularity_group", "unknown")).strip().lower() or "unknown"
        candidate_scores = _normalize_candidate_scores(record.get("candidate_scores"))
        if not candidate_scores:
            continue

        sorted_scores = sorted(candidate_scores, key=lambda row: row["score"], reverse=True)
        score_values = [row["score"] for row in sorted_scores]
        probabilities = _softmax(score_values)
        entropy = _normalized_entropy(probabilities)
        score_sharpness = 1.0 - entropy if not pd.isna(entropy) else float("nan")

        for rank, (row, probability) in enumerate(zip(sorted_scores, probabilities), start=1):
            rows.append(
                {
                    "task_type": "candidate_ranking",
                    "user_id": user_id,
                    "target_item_id": target_item_id,
                    "candidate_item_id": row["item_id"],
                    "label": int(row["item_id"] == target_item_id),
                    "rank": rank,
                    "raw_score": _safe_float(row["score"]),
                    "score_probability": _safe_float(probability),
                    "inverse_probability": _safe_float(1.0 - probability),
                    "normalized_entropy": _safe_float(entropy),
                    "score_sharpness": _safe_float(score_sharpness),
                    "target_popularity_group": popularity_group,
                }
            )

    return pd.DataFrame(rows)


def _add_pointwise_estimator_columns(out: pd.DataFrame, fused_alpha: float) -> pd.DataFrame:
    if "confidence" in out.columns:
        out["verbalized_confidence"] = add_verbalized_confidence(out)
        out["verbalized_uncertainty"] = 1.0 - out["verbalized_confidence"]

    if "calibrated_confidence" in out.columns:
        out["verbalized_calibrated_confidence"] = add_calibrated_confidence(out)
        if "uncertainty" in out.columns:
            out["verbalized_calibrated_uncertainty"] = _clip01(out["uncertainty"])
        else:
            out["verbalized_calibrated_uncertainty"] = 1.0 - out["verbalized_calibrated_confidence"]

    if "consistency_confidence" in out.columns:
        out["consistency_confidence"] = add_consistency_confidence(out)
        if "consistency_uncertainty" in out.columns:
            out["consistency_uncertainty"] = _clip01(out["consistency_uncertainty"])

    if {
        "calibrated_confidence",
        "uncertainty",
        "consistency_confidence",
        "consistency_uncertainty",
    }.issubset(out.columns):
        out["fused_confidence"] = add_fused_confidence(out, alpha=fused_alpha)
        out["fused_uncertainty"] = add_fused_uncertainty(out, alpha=fused_alpha)
        out["fused_alpha"] = fused_alpha

    return out


def _add_ranking_proxy_estimator_columns(out: pd.DataFrame) -> pd.DataFrame:
    if "score_margin" in out.columns:
        margin_conf = 1.0 / (1.0 + (-out["score_margin"].astype(float)).apply(math.exp))
        out["score_margin_confidence"] = _clip01(margin_conf)
        out["score_margin_uncertainty"] = 1.0 - out["score_margin_confidence"]

    if "score_entropy" in out.columns:
        out["score_entropy_uncertainty"] = _clip01(out["score_entropy"])
        out["score_entropy_confidence"] = 1.0 - out["score_entropy_uncertainty"]

    return out


def ensure_estimator_columns(df: pd.DataFrame, fused_alpha: float = 0.5) -> pd.DataFrame:
    out = df.copy()
    out = _add_pointwise_estimator_columns(out, fused_alpha=fused_alpha)
    out = _add_ranking_proxy_estimator_columns(out)
    return out


def get_available_estimators(df: pd.DataFrame, fused_alpha: float = 0.5) -> dict[str, dict[str, str | float]]:
    estimators: dict[str, dict[str, str | float]] = {}

    if {"verbalized_confidence", "verbalized_uncertainty"}.issubset(df.columns):
        estimators["verbalized_raw"] = {
            "confidence_col": "verbalized_confidence",
            "uncertainty_col": "verbalized_uncertainty",
            "task_family": "pointwise",
        }

    if {"verbalized_calibrated_confidence", "verbalized_calibrated_uncertainty"}.issubset(df.columns):
        estimators["verbalized_calibrated"] = {
            "confidence_col": "verbalized_calibrated_confidence",
            "uncertainty_col": "verbalized_calibrated_uncertainty",
            "task_family": "pointwise",
        }

    if {"consistency_confidence", "consistency_uncertainty"}.issubset(df.columns):
        estimators["consistency"] = {
            "confidence_col": "consistency_confidence",
            "uncertainty_col": "consistency_uncertainty",
            "task_family": "pointwise",
        }

    if {"fused_confidence", "fused_uncertainty"}.issubset(df.columns):
        estimators["fused"] = {
            "confidence_col": "fused_confidence",
            "uncertainty_col": "fused_uncertainty",
            "fusion_alpha": _normalize_alpha(fused_alpha),
            "task_family": "pointwise",
        }

    if {"score_margin_confidence", "score_margin_uncertainty"}.issubset(df.columns):
        estimators["score_margin"] = {
            "confidence_col": "score_margin_confidence",
            "uncertainty_col": "score_margin_uncertainty",
            "task_family": "ranking_proxy",
        }

    if {"score_entropy_confidence", "score_entropy_uncertainty"}.issubset(df.columns):
        estimators["score_entropy"] = {
            "confidence_col": "score_entropy_confidence",
            "uncertainty_col": "score_entropy_uncertainty",
            "task_family": "ranking_proxy",
        }

    return estimators

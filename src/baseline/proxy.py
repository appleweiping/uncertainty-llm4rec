from __future__ import annotations

import math

import pandas as pd


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    if total <= 0:
        return [float("nan")] * len(values)
    return [value / total for value in exps]


def _normalized_entropy(probabilities: list[float]) -> float:
    if not probabilities:
        return float("nan")
    valid_probs = [prob for prob in probabilities if prob > 0]
    if not valid_probs:
        return 0.0
    entropy = -sum(prob * math.log(prob + 1e-12) for prob in valid_probs)
    normalizer = math.log(len(probabilities) + 1e-12)
    if normalizer <= 0:
        return 0.0
    return float(entropy / normalizer)


def build_proxy_results(ranked_rows: pd.DataFrame) -> pd.DataFrame:
    if ranked_rows.empty:
        return pd.DataFrame(
            columns=[
                "user_id",
                "item_id",
                "label",
                "rank",
                "raw_score",
                "top1_score",
                "top2_score",
                "score_margin",
                "score_entropy",
                "rank_gap",
                "proxy_confidence",
                "target_popularity_group",
            ]
        )

    rows: list[dict[str, object]] = []
    for user_id, user_df in ranked_rows.groupby("user_id"):
        user_df = user_df.sort_values("rank").copy()
        raw_scores = user_df["score"].astype(float).tolist()
        has_full_scores = user_df["score"].notna().all()
        candidate_probs = _softmax(raw_scores) if has_full_scores else [float("nan")] * len(user_df)
        top1_score = float(user_df.iloc[0]["score"]) if user_df.iloc[0]["score"] == user_df.iloc[0]["score"] else float("nan")
        top2_score = (
            float(user_df.iloc[1]["score"])
            if len(user_df) >= 2 and user_df.iloc[1]["score"] == user_df.iloc[1]["score"]
            else float("nan")
        )
        score_margin = top1_score - top2_score if top1_score == top1_score and top2_score == top2_score else float("nan")
        score_entropy = _normalized_entropy(candidate_probs) if has_full_scores else float("nan")

        for (_, row), candidate_prob in zip(user_df.iterrows(), candidate_probs):
            rank = int(row["rank"])
            rows.append(
                {
                    "user_id": str(user_id),
                    "item_id": str(row["candidate_item_id"]),
                    "label": int(row["label"]),
                    "rank": rank,
                    "raw_score": float(row["score"]) if row["score"] == row["score"] else float("nan"),
                    "top1_score": top1_score,
                    "top2_score": top2_score,
                    "score_margin": score_margin,
                    "score_entropy": score_entropy,
                    "rank_gap": int(rank - 1),
                    "proxy_confidence": float(candidate_prob) if candidate_prob == candidate_prob else float("nan"),
                    "target_popularity_group": str(row.get("target_popularity_group", "unknown")).strip().lower() or "unknown",
                }
            )

    return pd.DataFrame(rows)

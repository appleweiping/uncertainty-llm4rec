from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.eval.ranking_metrics import compute_ranking_metrics


def build_ranked_rows(
    grouped_samples: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> pd.DataFrame:
    sample_by_user = {
        str(sample.get("user_id", "")).strip(): sample
        for sample in grouped_samples
    }
    rows: list[dict[str, Any]] = []

    for prediction in predictions:
        user_id = str(prediction.get("user_id", "")).strip()
        sample = sample_by_user.get(user_id, {})
        candidates = sample.get("candidates", [])
        candidate_by_id = {
            str(candidate.get("item_id", "")).strip(): candidate
            for candidate in candidates
        }
        candidate_item_ids = [str(item_id).strip() for item_id in prediction.get("candidate_item_ids", [])]
        ranked_item_ids = [str(item_id).strip() for item_id in prediction.get("ranked_item_ids", [])]
        score_by_item = {}
        scores = prediction.get("scores")
        if isinstance(scores, list) and len(scores) == len(candidate_item_ids):
            score_by_item = {
                item_id: float(score)
                for item_id, score in zip(candidate_item_ids, scores)
            }

        for rank, item_id in enumerate(ranked_item_ids, start=1):
            candidate = candidate_by_id.get(item_id, {})
            rows.append(
                {
                    "user_id": user_id,
                    "candidate_item_id": item_id,
                    "label": int(candidate.get("label", 0)),
                    "score": score_by_item.get(item_id, float("nan")),
                    "rank": rank,
                    "target_popularity_group": str(
                        prediction.get("metadata", {}).get("target_popularity_group")
                        or sample.get("target_popularity_group", "unknown")
                    ).strip().lower()
                    or "unknown",
                }
            )

    return pd.DataFrame(rows)


def build_score_rows(ranked_rows: pd.DataFrame) -> pd.DataFrame:
    if ranked_rows.empty:
        return ranked_rows.copy()
    columns = ["user_id", "candidate_item_id", "label", "score", "rank", "target_popularity_group"]
    return ranked_rows[columns].copy()


def _recall_for_user(user_df: pd.DataFrame, k: int) -> float:
    topk = user_df.nsmallest(k, "rank")
    total_positives = int(user_df["label"].astype(int).sum())
    if total_positives <= 0:
        return 0.0
    hits = int(topk["label"].astype(int).sum())
    return float(hits / total_positives)


def compute_nh_nr_metrics(ranked_rows: pd.DataFrame, k: int = 10) -> dict[str, float]:
    if ranked_rows.empty:
        return {
            f"HR@{k}": float("nan"),
            f"NDCG@{k}": float("nan"),
            f"Recall@{k}": float("nan"),
            f"MRR@{k}": float("nan"),
            "num_users": 0,
            "num_rows": 0,
        }

    nh_metrics = compute_ranking_metrics(ranked_rows, k=k)
    recalls = []
    for _, user_df in ranked_rows.groupby("user_id"):
        recalls.append(_recall_for_user(user_df, k=k))

    return {
        **nh_metrics,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else float("nan"),
        "num_rows": int(len(ranked_rows)),
    }

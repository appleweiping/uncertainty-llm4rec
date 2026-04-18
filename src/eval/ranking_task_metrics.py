from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _normalize_item_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [text]


def build_ranking_eval_frame(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for record in predictions_df.to_dict(orient="records"):
        candidate_ids = _normalize_item_list(record.get("candidate_item_ids"))
        ranked_ids = _normalize_item_list(record.get("pred_ranked_item_ids"))
        topk_ids = _normalize_item_list(record.get("topk_item_ids"))
        popularity_groups = _normalize_item_list(record.get("candidate_popularity_groups"))

        candidate_popularity = {
            str(item_id): (
                str(popularity_groups[idx]).strip().lower()
                if idx < len(popularity_groups) and str(popularity_groups[idx]).strip()
                else "unknown"
            )
            for idx, item_id in enumerate(candidate_ids)
        }

        positive_item_id = str(record.get("positive_item_id", "")).strip()
        parse_success = bool(record.get("parse_success", False))

        if positive_item_id and positive_item_id in ranked_ids:
            positive_rank = ranked_ids.index(positive_item_id) + 1
        else:
            positive_rank = len(ranked_ids) + 1 if ranked_ids else len(candidate_ids) + 1

        rows.append(
            {
                "user_id": record.get("user_id"),
                "source_event_id": record.get("source_event_id"),
                "split_name": record.get("split_name"),
                "timestamp": record.get("timestamp"),
                "positive_item_id": positive_item_id,
                "positive_rank": int(positive_rank),
                "num_candidates": int(len(candidate_ids)),
                "candidate_item_ids": candidate_ids,
                "candidate_popularity_groups": [candidate_popularity.get(item_id, "unknown") for item_id in candidate_ids],
                "pred_ranked_item_ids": ranked_ids,
                "topk_item_ids": topk_ids,
                "positive_in_candidate": positive_item_id in candidate_ids if positive_item_id else False,
                "positive_in_prediction": positive_item_id in ranked_ids if positive_item_id else False,
                "positive_popularity_group": candidate_popularity.get(positive_item_id, "unknown"),
                "topk_popularity_groups": [candidate_popularity.get(item_id, "unknown") for item_id in topk_ids],
                "parse_success": parse_success,
                "latency": float(record.get("latency", np.nan)),
                "confidence": float(record.get("confidence", np.nan)),
                "contains_out_of_candidate_item": bool(record.get("contains_out_of_candidate_item", False)),
                "raw_response": record.get("raw_response"),
            }
        )

    return pd.DataFrame(rows)


def compute_ranking_task_metrics(
    ranking_eval_df: pd.DataFrame,
    *,
    k: int = 10,
) -> dict[str, float]:
    if ranking_eval_df.empty:
        return {
            "sample_count": 0,
            "parse_success_rate": float("nan"),
            "avg_latency": float("nan"),
            "avg_confidence": float("nan"),
            "avg_candidates": float("nan"),
            f"HR@{k}": float("nan"),
            f"NDCG@{k}": float("nan"),
            "MRR": float("nan"),
            f"coverage@{k}": float("nan"),
            f"head_exposure_ratio@{k}": float("nan"),
            f"longtail_coverage@{k}": float("nan"),
            "out_of_candidate_rate": float("nan"),
        }

    df = ranking_eval_df.copy()

    topk_hits = (df["positive_rank"] <= k).astype(float)
    ndcg = df["positive_rank"].apply(lambda rank: float(1.0 / np.log2(rank + 1)) if rank <= k else 0.0)
    mrr = df["positive_rank"].apply(lambda rank: float(1.0 / rank) if rank > 0 else 0.0)

    exposed_rows: list[dict[str, str]] = []
    all_tail_items: set[str] = set()
    all_candidate_items: set[str] = set()
    exposed_topk_items: set[str] = set()
    exposed_tail_items: set[str] = set()

    for record in df.to_dict(orient="records"):
        candidate_ids = _normalize_item_list(record.get("candidate_item_ids"))
        candidate_groups = _normalize_item_list(record.get("candidate_popularity_groups"))
        topk_ids = _normalize_item_list(record.get("topk_item_ids"))
        topk_groups = _normalize_item_list(record.get("topk_popularity_groups"))

        all_candidate_items.update(candidate_ids)
        for idx, item_id in enumerate(candidate_ids):
            popularity = candidate_groups[idx] if idx < len(candidate_groups) else "unknown"
            popularity = str(popularity).strip().lower()
            if popularity == "tail":
                all_tail_items.add(item_id)

        for idx, item_id in enumerate(topk_ids):
            popularity = topk_groups[idx] if idx < len(topk_groups) else "unknown"
            popularity = str(popularity).strip().lower()
            exposed_rows.append({"item_id": str(item_id), "popularity_group": popularity})
            exposed_topk_items.add(str(item_id))
            if popularity == "tail":
                exposed_tail_items.add(str(item_id))

    exposed_df = pd.DataFrame(exposed_rows)

    if exposed_df.empty:
        head_exposure_ratio = float("nan")
    else:
        head_exposure_ratio = float((exposed_df["popularity_group"] == "head").mean())

    coverage = float(len(exposed_topk_items) / len(all_candidate_items)) if all_candidate_items else float("nan")
    longtail_coverage = float(len(exposed_tail_items) / len(all_tail_items)) if all_tail_items else 0.0

    return {
        "sample_count": int(len(df)),
        "parse_success_rate": float(df["parse_success"].mean()),
        "avg_latency": float(df["latency"].mean()),
        "avg_confidence": float(df["confidence"].mean()),
        "avg_candidates": float(df["num_candidates"].mean()),
        f"HR@{k}": float(topk_hits.mean()),
        f"NDCG@{k}": float(ndcg.mean()),
        "MRR": float(mrr.mean()),
        f"coverage@{k}": coverage,
        f"head_exposure_ratio@{k}": head_exposure_ratio,
        f"longtail_coverage@{k}": longtail_coverage,
        "out_of_candidate_rate": float(df["contains_out_of_candidate_item"].mean()),
    }


def compute_ranking_exposure_distribution(
    ranking_eval_df: pd.DataFrame,
    *,
    k: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for record in ranking_eval_df.to_dict(orient="records"):
        topk_ids = _normalize_item_list(record.get("topk_item_ids"))[:k]
        topk_groups = _normalize_item_list(record.get("topk_popularity_groups"))[:k]
        for idx, item_id in enumerate(topk_ids):
            popularity_group = topk_groups[idx] if idx < len(topk_groups) else "unknown"
            rows.append(
                {
                    "item_id": str(item_id),
                    "popularity_group": str(popularity_group).strip().lower(),
                }
            )

    exposure_df = pd.DataFrame(rows)
    if exposure_df.empty:
        return pd.DataFrame(columns=["popularity_group", "fraction"])

    return (
        exposure_df["popularity_group"]
        .value_counts(normalize=True)
        .rename("fraction")
        .reset_index()
        .rename(columns={"index": "popularity_group"})
    )

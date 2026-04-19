from __future__ import annotations

from typing import Any


POPULARITY_PRIORITY = {
    "head": 0,
    "mid": 1,
    "tail": 2,
    "unknown": 3,
}


def build_popularity_prior_predictions(samples: list[dict[str, Any]], *, k: int = 10) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for sample in samples:
        candidate_ids = [str(item_id) for item_id in sample.get("candidate_item_ids", [])]
        popularity_groups = [str(group).lower() for group in sample.get("candidate_popularity_groups", [])]
        group_by_item = {
            item_id: popularity_groups[idx] if idx < len(popularity_groups) else "unknown"
            for idx, item_id in enumerate(candidate_ids)
        }

        ranked_ids = sorted(
            candidate_ids,
            key=lambda item_id: (POPULARITY_PRIORITY.get(group_by_item.get(item_id, "unknown"), 3), candidate_ids.index(item_id)),
        )
        predictions.append(
            {
                "user_id": sample.get("user_id"),
                "source_event_id": sample.get("source_event_id"),
                "split_name": sample.get("split_name"),
                "timestamp": sample.get("timestamp"),
                "positive_item_id": sample.get("positive_item_id"),
                "candidate_item_ids": candidate_ids,
                "candidate_popularity_groups": popularity_groups,
                "pred_ranked_item_ids": ranked_ids,
                "topk_item_ids": ranked_ids[:k],
                "parse_success": True,
                "latency": 0.0,
                "confidence": 1.0,
                "contains_out_of_candidate_item": False,
                "raw_response": "popularity_prior_rank",
            }
        )
    return predictions


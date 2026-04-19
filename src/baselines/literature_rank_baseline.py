from __future__ import annotations

import math
import re
from typing import Any


POPULARITY_PRIORITY = {
    "head": 0,
    "mid": 1,
    "tail": 2,
    "unknown": 3,
}

LONGTAIL_PRIORITY = {
    "tail": 0,
    "mid": 1,
    "head": 2,
    "unknown": 3,
}


def build_candidate_order_predictions(samples: list[dict[str, Any]], *, k: int = 10) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for sample in samples:
        candidate_ids = [str(item_id) for item_id in sample.get("candidate_item_ids", [])]
        popularity_groups = _candidate_popularity_groups(sample, candidate_ids)
        ranked_ids = list(candidate_ids)
        predictions.append(_prediction_record(sample, candidate_ids, popularity_groups, ranked_ids, "candidate_order_rank", k))
    return predictions


def build_popularity_prior_predictions(samples: list[dict[str, Any]], *, k: int = 10) -> list[dict[str, Any]]:
    return _build_group_prior_predictions(samples, priority=POPULARITY_PRIORITY, baseline_name="popularity_prior_rank", k=k)


def build_longtail_prior_predictions(samples: list[dict[str, Any]], *, k: int = 10) -> list[dict[str, Any]]:
    return _build_group_prior_predictions(samples, priority=LONGTAIL_PRIORITY, baseline_name="longtail_prior_rank", k=k)


def build_history_overlap_predictions(samples: list[dict[str, Any]], *, k: int = 10) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for sample in samples:
        candidate_ids = [str(item_id) for item_id in sample.get("candidate_item_ids", [])]
        popularity_groups = _candidate_popularity_groups(sample, candidate_ids)
        candidate_texts = [str(text) for text in sample.get("candidate_texts", [])]
        candidate_titles = [str(text) for text in sample.get("candidate_titles", [])]
        history_tokens = _tokens(" ".join(str(item) for item in sample.get("history", [])))
        scores: dict[str, float] = {}
        for idx, item_id in enumerate(candidate_ids):
            text = candidate_texts[idx] if idx < len(candidate_texts) else ""
            if not text and idx < len(candidate_titles):
                text = candidate_titles[idx]
            item_tokens = _tokens(text)
            scores[item_id] = _jaccard(history_tokens, item_tokens)

        ranked_ids = sorted(
            candidate_ids,
            key=lambda item_id: (-scores.get(item_id, 0.0), candidate_ids.index(item_id)),
        )
        predictions.append(
            _prediction_record(
                sample,
                candidate_ids,
                popularity_groups,
                ranked_ids,
                "history_overlap_rank",
                k,
                candidate_scores=scores,
            )
        )
    return predictions


def _build_group_prior_predictions(
    samples: list[dict[str, Any]],
    *,
    priority: dict[str, int],
    baseline_name: str,
    k: int,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for sample in samples:
        candidate_ids = [str(item_id) for item_id in sample.get("candidate_item_ids", [])]
        popularity_groups = _candidate_popularity_groups(sample, candidate_ids)
        group_by_item = {
            item_id: popularity_groups[idx] if idx < len(popularity_groups) else "unknown"
            for idx, item_id in enumerate(candidate_ids)
        }
        ranked_ids = sorted(
            candidate_ids,
            key=lambda item_id: (priority.get(group_by_item.get(item_id, "unknown"), 3), candidate_ids.index(item_id)),
        )
        predictions.append(_prediction_record(sample, candidate_ids, popularity_groups, ranked_ids, baseline_name, k))
    return predictions


def _candidate_popularity_groups(sample: dict[str, Any], candidate_ids: list[str]) -> list[str]:
    groups = [str(group).lower() for group in sample.get("candidate_popularity_groups", [])]
    if len(groups) < len(candidate_ids):
        groups.extend(["unknown"] * (len(candidate_ids) - len(groups)))
    return groups[: len(candidate_ids)]


def _prediction_record(
    sample: dict[str, Any],
    candidate_ids: list[str],
    popularity_groups: list[str],
    ranked_ids: list[str],
    baseline_name: str,
    k: int,
    *,
    candidate_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "user_id": sample.get("user_id"),
        "source_event_id": sample.get("source_event_id"),
        "split_name": sample.get("split_name"),
        "timestamp": sample.get("timestamp"),
        "positive_item_id": sample.get("positive_item_id"),
        "candidate_item_ids": candidate_ids,
        "candidate_titles": sample.get("candidate_titles", []),
        "candidate_texts": sample.get("candidate_texts", []),
        "candidate_popularity_groups": popularity_groups,
        "pred_ranked_item_ids": ranked_ids,
        "topk_item_ids": ranked_ids[:k],
        "parse_success": True,
        "latency": 0.0,
        "confidence": 1.0,
        "contains_out_of_candidate_item": False,
        "raw_response": baseline_name,
        "candidate_scores": candidate_scores or {},
    }


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    denominator = len(left | right)
    if denominator == 0:
        return 0.0
    score = len(left & right) / denominator
    return float(score) if math.isfinite(score) else 0.0


BASELINE_BUILDERS = {
    "candidate_order_rank": build_candidate_order_predictions,
    "popularity_prior_rank": build_popularity_prior_predictions,
    "longtail_prior_rank": build_longtail_prior_predictions,
    "history_overlap_rank": build_history_overlap_predictions,
}


BASELINE_NOTES = {
    "candidate_order_rank": "Task-aligned candidate-order baseline; mirrors the candidate list prior without uncertainty.",
    "popularity_prior_rank": "Popularity-prior ranking baseline aligned with common recommendation popularity controls.",
    "longtail_prior_rank": "Long-tail-prior ranking baseline used to expose utility/exposure tradeoffs under the same candidate set.",
    "history_overlap_rank": "Lightweight content-similarity baseline inspired by unified LLM4Rec task-schema evaluations; no uncertainty is used.",
}


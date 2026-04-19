from __future__ import annotations

import re
from typing import Any


POPULARITY_PRIORITY = {
    "head": 0,
    "mid": 1,
    "tail": 2,
    "unknown": 3,
}


def build_history_overlap_pairwise_predictions(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for sample in samples:
        item_a_id = str(sample.get("item_a_id", "")).strip()
        item_b_id = str(sample.get("item_b_id", "")).strip()
        history_tokens = _tokens(" ".join(str(item) for item in sample.get("history", [])))
        score_a = _jaccard(history_tokens, _tokens(str(sample.get("item_a_text") or sample.get("item_a_title") or "")))
        score_b = _jaccard(history_tokens, _tokens(str(sample.get("item_b_text") or sample.get("item_b_title") or "")))
        if score_a == score_b:
            preferred_item = item_a_id
        else:
            preferred_item = item_a_id if score_a > score_b else item_b_id
        confidence = 0.5 + min(abs(score_a - score_b), 0.5)
        predictions.append(
            _prediction_record(
                sample,
                preferred_item,
                "history_overlap_pairwise",
                confidence=confidence,
                scores={"item_a_score": score_a, "item_b_score": score_b},
            )
        )
    return predictions


def build_popularity_prior_pairwise_predictions(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for sample in samples:
        item_a_id = str(sample.get("item_a_id", "")).strip()
        item_b_id = str(sample.get("item_b_id", "")).strip()
        group_a = str(sample.get("item_a_popularity_group", "unknown")).strip().lower()
        group_b = str(sample.get("item_b_popularity_group", "unknown")).strip().lower()
        priority_a = POPULARITY_PRIORITY.get(group_a, 3)
        priority_b = POPULARITY_PRIORITY.get(group_b, 3)
        preferred_item = item_a_id if priority_a <= priority_b else item_b_id
        confidence = 0.65 if priority_a != priority_b else 0.5
        predictions.append(
            _prediction_record(
                sample,
                preferred_item,
                "popularity_prior_pairwise",
                confidence=confidence,
                scores={"item_a_priority": priority_a, "item_b_priority": priority_b},
            )
        )
    return predictions


def _prediction_record(
    sample: dict[str, Any],
    preferred_item: str,
    baseline_name: str,
    *,
    confidence: float,
    scores: dict[str, float],
) -> dict[str, Any]:
    return {
        "pair_id": sample.get("pair_id"),
        "source_event_id": sample.get("source_event_id"),
        "user_id": sample.get("user_id"),
        "item_a_id": sample.get("item_a_id"),
        "item_b_id": sample.get("item_b_id"),
        "preferred_item_true": sample.get("preferred_item") or sample.get("preferred_item_true"),
        "preferred_item_pred": preferred_item,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "reason": baseline_name,
        "pair_type": sample.get("pair_type"),
        "split_name": sample.get("split_name"),
        "timestamp": sample.get("timestamp"),
        "parse_mode": "baseline",
        "parse_success": True,
        "ambiguous_preference": False,
        "latency": 0.0,
        "model_name": baseline_name,
        "provider": "literature_aligned_baseline",
        "raw_response": baseline_name,
        "candidate_scores": scores,
    }


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return float(len(left & right) / len(left | right))


PAIRWISE_BASELINE_BUILDERS = {
    "history_overlap_pairwise": build_history_overlap_pairwise_predictions,
    "popularity_prior_pairwise": build_popularity_prior_pairwise_predictions,
}


PAIRWISE_BASELINE_NOTES = {
    "history_overlap_pairwise": "Pairwise content-overlap preference baseline aligned with local preference evaluation.",
    "popularity_prior_pairwise": "Pairwise popularity-prior preference baseline over the same pair construction.",
}


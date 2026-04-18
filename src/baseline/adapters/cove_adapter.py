from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from src.baseline.base import BaselineAdapter


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


def _stable_bias(item_id: str) -> float:
    digest = hashlib.md5(item_id.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 1000) / 1_000_000.0


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    if total <= 0:
        return [0.0 for _ in values]
    return [value / total for value in exps]


class CoVEAdapter(BaselineAdapter):
    def __init__(self) -> None:
        super().__init__(baseline_name="cove")

    def predict_group(self, grouped_sample: dict[str, Any]) -> dict[str, Any]:
        history_text = " ".join(grouped_sample.get("history") or grouped_sample.get("history_items") or [])
        history_tokens = set(_tokenize(history_text))
        candidates = grouped_sample.get("candidates", [])

        logits: list[float] = []
        candidate_item_ids: list[str] = []
        item_tokens: dict[str, str] = {}
        for index, candidate in enumerate(candidates, start=1):
            item_id = str(candidate.get("item_id", "")).strip()
            title = str(candidate.get("title", "")).strip()
            meta = str(candidate.get("meta", "")).strip()
            candidate_tokens = _tokenize(f"{title} {meta}")
            candidate_token_set = set(candidate_tokens)

            overlap = len(history_tokens & candidate_token_set)
            normalized_overlap = overlap / max(len(history_tokens), 1)
            token_density = len(candidate_token_set) / max(len(candidate_tokens), 1)
            raw_logit = normalized_overlap + 0.05 * token_density + _stable_bias(item_id)

            candidate_item_ids.append(item_id)
            logits.append(raw_logit)
            item_tokens[item_id] = f"<ITEM_{index}>"

        scores = _softmax(logits)
        sorted_pairs = sorted(zip(candidate_item_ids, scores), key=lambda pair: pair[1], reverse=True)
        ranked_item_ids = [item_id for item_id, _ in sorted_pairs]
        score_by_item = {item_id: score for item_id, score in zip(candidate_item_ids, scores)}

        return {
            "user_id": str(grouped_sample.get("user_id", "")).strip(),
            "candidate_item_ids": candidate_item_ids,
            "scores": [score_by_item[item_id] for item_id in candidate_item_ids],
            "ranked_item_ids": ranked_item_ids,
            "metadata": {
                "baseline_name": self.baseline_name,
                "scorer_type": "cove_style_candidate_softmax",
                "placeholder": True,
                "target_item_id": grouped_sample.get("target_item_id", ""),
                "target_popularity_group": grouped_sample.get("target_popularity_group", "unknown"),
                "item_tokens": item_tokens,
            },
        }

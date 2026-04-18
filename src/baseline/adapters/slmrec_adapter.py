from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any

import numpy as np

from src.baseline.base import BaselineAdapter


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


def _stable_bias(item_id: str) -> float:
    digest = hashlib.md5(item_id.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 1000) / 1_000_000.0


class SLMRecAdapter(BaselineAdapter):
    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__(baseline_name="slmrec")
        self.embedding_dim = embedding_dim
        self.idf: dict[str, float] = {}

    def fit(self, train_data: list[dict[str, Any]], valid_data: list[dict[str, Any]] | None = None) -> None:
        documents: list[set[str]] = []
        for sample in train_data + (valid_data or []):
            history_docs = sample.get("history") or sample.get("history_items") or []
            if history_docs:
                documents.append(set(_tokenize(" ".join(str(item) for item in history_docs))))
            for candidate in sample.get("candidates", []):
                documents.append(set(_tokenize(f"{candidate.get('title', '')} {candidate.get('meta', '')}")))

        doc_freq: Counter[str] = Counter()
        for doc_tokens in documents:
            doc_freq.update(token for token in doc_tokens if token)

        num_docs = max(len(documents), 1)
        self.idf = {
            token: math.log((1 + num_docs) / (1 + freq)) + 1.0
            for token, freq in doc_freq.items()
        }

    def _embed_text(self, text: str) -> np.ndarray:
        vector = np.zeros(self.embedding_dim, dtype=float)
        tokens = _tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.embedding_dim
            sign = 1.0 if int(digest[8:16], 16) % 2 == 0 else -1.0
            weight = self.idf.get(token, 1.0)
            vector[index] += sign * weight
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def _cosine(self, left: np.ndarray, right: np.ndarray) -> float:
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm <= 0 or right_norm <= 0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))

    def predict_group(self, grouped_sample: dict[str, Any]) -> dict[str, Any]:
        history = grouped_sample.get("history") or grouped_sample.get("history_items") or []
        history_text = " ".join(str(item) for item in history)
        user_repr = self._embed_text(history_text)

        candidate_item_ids: list[str] = []
        scores: list[float] = []
        for candidate in grouped_sample.get("candidates", []):
            item_id = str(candidate.get("item_id", "")).strip()
            candidate_text = f"{candidate.get('title', '')} {candidate.get('meta', '')}".strip()
            item_repr = self._embed_text(candidate_text)
            score = self._cosine(user_repr, item_repr) + _stable_bias(item_id)
            candidate_item_ids.append(item_id)
            scores.append(score)

        sorted_pairs = sorted(zip(candidate_item_ids, scores), key=lambda pair: pair[1], reverse=True)
        ranked_item_ids = [item_id for item_id, _ in sorted_pairs]

        return {
            "user_id": str(grouped_sample.get("user_id", "")).strip(),
            "candidate_item_ids": candidate_item_ids,
            "scores": scores,
            "ranked_item_ids": ranked_item_ids,
            "metadata": {
                "baseline_name": self.baseline_name,
                "scorer_type": "embedding_similarity",
                "embedding_dim": self.embedding_dim,
                "target_item_id": grouped_sample.get("target_item_id", ""),
                "target_popularity_group": grouped_sample.get("target_popularity_group", "unknown"),
            },
        }

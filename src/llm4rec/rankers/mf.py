"""Pure-Python minimal matrix-factorization ranker for smoke baselines."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any

from llm4rec.rankers.base import CheckpointNotImplementedMixin, RankingResult, prediction_from_scores


class MatrixFactorizationRanker(CheckpointNotImplementedMixin):
    """A lightweight implicit-feedback MF baseline, not a full BPR implementation."""

    method_name = "mf"

    def __init__(
        self,
        *,
        seed: int = 0,
        factors: int = 8,
        epochs: int = 25,
        learning_rate: float = 0.05,
        regularization: float = 0.001,
    ) -> None:
        self.seed = int(seed)
        self.factors = int(factors)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.user_factors: dict[str, list[float]] = {}
        self.item_factors: dict[str, list[float]] = {}
        self.user_bias: defaultdict[str, float] = defaultdict(float)
        self.item_bias: defaultdict[str, float] = defaultdict(float)

    def fit(
        self,
        train_examples: list[dict[str, Any]],
        item_catalog: list[dict[str, Any]],
        interactions: list[dict[str, Any]] | None = None,
    ) -> None:
        item_ids = sorted(str(row["item_id"]) for row in item_catalog)
        users = sorted({str(example["user_id"]) for example in train_examples})
        self.user_factors = {user_id: self._init_vector(f"user:{user_id}") for user_id in users}
        self.item_factors = {item_id: self._init_vector(f"item:{item_id}") for item_id in item_ids}
        positives = _positive_pairs(train_examples)
        if not positives:
            return
        for _ in range(max(self.epochs, 0)):
            for user_id, item_id in positives:
                self._sgd_step(user_id, item_id, target=1.0)

    def rank(self, example: dict[str, Any], candidate_items: list[str]) -> RankingResult:
        user_id = str(example["user_id"])
        scores = {str(item_id): self._score(user_id, str(item_id)) for item_id in candidate_items}
        return prediction_from_scores(
            example=example,
            candidate_items=candidate_items,
            item_scores=scores,
            method=self.method_name,
            metadata={
                "training_objective": "implicit_positive_mf_squared_error",
                "seed": self.seed,
                "factors": self.factors,
                "epochs": self.epochs,
                "label_leakage": False,
                "not_bpr": True,
            },
        )

    def _sgd_step(self, user_id: str, item_id: str, *, target: float) -> None:
        user = self.user_factors.setdefault(user_id, self._init_vector(f"user:{user_id}"))
        item = self.item_factors.setdefault(item_id, self._init_vector(f"item:{item_id}"))
        pred = self._dot(user, item) + self.user_bias[user_id] + self.item_bias[item_id]
        err = target - pred
        lr = self.learning_rate
        reg = self.regularization
        self.user_bias[user_id] += lr * (err - reg * self.user_bias[user_id])
        self.item_bias[item_id] += lr * (err - reg * self.item_bias[item_id])
        for index in range(self.factors):
            u_old = user[index]
            i_old = item[index]
            user[index] += lr * (err * i_old - reg * u_old)
            item[index] += lr * (err * u_old - reg * i_old)

    def _score(self, user_id: str, item_id: str) -> float:
        user = self.user_factors.get(user_id)
        item = self.item_factors.get(item_id)
        if user is None or item is None:
            return float(self.item_bias.get(item_id, 0.0))
        return self._dot(user, item) + self.user_bias[user_id] + self.item_bias[item_id]

    def _init_vector(self, key: str) -> list[float]:
        rng = random.Random(_stable_seed(self.seed, key))
        return [rng.uniform(-0.01, 0.01) for _ in range(self.factors)]

    @staticmethod
    def _dot(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))


def _positive_pairs(train_examples: list[dict[str, Any]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for example in train_examples:
        user_id = str(example["user_id"])
        for item_id in [*example.get("history", []), example["target"]]:
            pair = (user_id, str(item_id))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def _stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)

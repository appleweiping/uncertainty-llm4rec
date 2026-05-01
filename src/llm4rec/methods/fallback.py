"""Fallback ranker routing for Phase 6 OursMethod."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm4rec.rankers.base import BaseRanker, RankingResult
from llm4rec.rankers.bm25 import BM25Ranker
from llm4rec.rankers.popularity import PopularityRanker
from llm4rec.rankers.sequential import MarkovSequentialRanker

SUPPORTED_FALLBACKS = {"bm25", "popularity", "sequential_markov"}


@dataclass(slots=True)
class FallbackRouter:
    method: str
    ranker: BaseRanker

    def fit(
        self,
        train_examples: list[dict[str, Any]],
        item_catalog: list[dict[str, Any]],
        interactions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.ranker.fit(train_examples, item_catalog, interactions)

    def rank(self, example: dict[str, Any], candidate_items: list[str]) -> RankingResult:
        return self.ranker.rank(example, [str(item_id) for item_id in candidate_items])


def build_fallback_router(method: str, params: dict[str, Any] | None = None) -> FallbackRouter:
    fallback_method = str(method or "bm25")
    fallback_params = dict(params or {})
    if fallback_method not in SUPPORTED_FALLBACKS:
        raise ValueError(
            f"unsupported OursMethod fallback: {fallback_method}; "
            f"supported={sorted(SUPPORTED_FALLBACKS)}"
        )
    if fallback_method == "bm25":
        ranker: BaseRanker = BM25Ranker(
            text_policy=str(fallback_params.get("text_policy") or "title"),
            k1=float(fallback_params.get("k1") or 1.5),
            b=float(fallback_params.get("b") or 0.75),
        )
    elif fallback_method == "popularity":
        ranker = PopularityRanker()
    else:
        ranker = MarkovSequentialRanker(
            max_history_length=int(fallback_params.get("max_history_length") or 50)
        )
    return FallbackRouter(method=fallback_method, ranker=ranker)

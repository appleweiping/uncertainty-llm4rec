"""BM25 retriever backed by the BM25 ranker."""

from __future__ import annotations

from typing import Any

from llm4rec.rankers.bm25 import BM25Ranker
from llm4rec.retrievers.base import RetrievalResult


class BM25Retriever:
    method_name = "bm25"

    def __init__(self, *, text_policy: str = "title") -> None:
        self.ranker = BM25Ranker(text_policy=text_policy)
        self.item_ids: list[str] = []

    def fit(
        self,
        train_examples: list[dict[str, Any]],
        item_catalog: list[dict[str, Any]],
        interactions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.item_ids = sorted(str(row["item_id"]) for row in item_catalog)
        self.ranker.fit(train_examples, item_catalog, interactions)

    def retrieve(self, example: dict[str, Any], k: int) -> RetrievalResult:
        result = self.ranker.rank(example, self.item_ids)
        return RetrievalResult(
            user_id=result.user_id,
            items=result.predicted_items[:k],
            scores=result.scores[:k],
            metadata={"method": self.method_name},
        )

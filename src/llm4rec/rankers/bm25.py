"""Small no-dependency BM25 ranker baseline."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from llm4rec.data.base import ItemRecord
from llm4rec.data.text_fields import item_text
from llm4rec.rankers.base import CheckpointNotImplementedMixin, RankingResult, prediction_from_scores

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class BM25Ranker(CheckpointNotImplementedMixin):
    method_name = "bm25"

    def __init__(self, *, text_policy: str = "title", k1: float = 1.5, b: float = 0.75) -> None:
        self.text_policy = text_policy
        self.k1 = float(k1)
        self.b = float(b)
        self.documents: dict[str, Counter[str]] = {}
        self.doc_lengths: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def fit(
        self,
        train_examples: list[dict[str, Any]],
        item_catalog: list[dict[str, Any]],
        interactions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.documents = {}
        df: Counter[str] = Counter()
        for row in item_catalog:
            item = ItemRecord(
                item_id=str(row["item_id"]),
                title=str(row.get("title") or row["item_id"]),
                description=str(row.get("description") or "") or None,
                category=str(row.get("category") or row.get("genres") or "") or None,
                domain=str(row.get("domain") or "") or None,
            )
            tokens = Counter(_tokenize(item_text(item, policy=self.text_policy)))
            self.documents[item.item_id] = tokens
            self.doc_lengths[item.item_id] = sum(tokens.values())
            for token in tokens:
                df[token] += 1
        doc_count = max(len(self.documents), 1)
        self.avgdl = (
            sum(self.doc_lengths.values()) / len(self.doc_lengths)
            if self.doc_lengths
            else 0.0
        )
        self.idf = {
            token: math.log(1.0 + (doc_count - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }
        self.title_by_id = {str(row["item_id"]): str(row.get("title") or "") for row in item_catalog}

    def rank(self, example: dict[str, Any], candidate_items: list[str]) -> RankingResult:
        query = _tokenize(" ".join(str(title) for title in _history_titles(example, self.title_by_id)))
        scores = {
            str(item_id): self._score(str(item_id), query)
            for item_id in candidate_items
        }
        return prediction_from_scores(
            example=example,
            candidate_items=candidate_items,
            item_scores=scores,
            method=self.method_name,
            metadata={
                "text_policy": self.text_policy,
                "query_source": "history_item_text_only",
                "label_leakage": False,
            },
        )

    def _score(self, item_id: str, query_tokens: list[str]) -> float:
        doc = self.documents.get(item_id, Counter())
        if not doc or not query_tokens:
            return 0.0
        dl = self.doc_lengths.get(item_id, 0)
        score = 0.0
        for token in query_tokens:
            tf = doc.get(token, 0)
            if tf <= 0:
                continue
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-12))
            score += self.idf.get(token, 0.0) * numerator / denominator
        return score


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _history_titles(example: dict[str, Any], title_by_id: dict[str, str]) -> list[str]:
    titles = example.get("history_titles")
    if isinstance(titles, list) and titles:
        return [str(title) for title in titles]
    return [title_by_id.get(str(item_id), str(item_id)) for item_id in example.get("history", [])]

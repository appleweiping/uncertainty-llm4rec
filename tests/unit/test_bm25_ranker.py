from __future__ import annotations

from llm4rec.rankers.bm25 import BM25Ranker
from llm4rec.retrievers.bm25 import BM25Retriever


def _catalog() -> list[dict[str, str]]:
    return [
        {"item_id": "i1", "title": "Space Odyssey"},
        {"item_id": "i2", "title": "Romantic Comedy"},
        {"item_id": "i3", "title": "Space Sequel"},
    ]


def test_bm25_ranker_uses_history_text_not_target_text() -> None:
    ranker = BM25Ranker(text_policy="title")
    ranker.fit([], _catalog())
    result = ranker.rank(
        {"example_id": "u:1", "user_id": "u", "history": ["i1"], "target": "i2", "split": "test", "domain": "tiny"},
        ["i2", "i3"],
    )
    assert result.predicted_items[0] == "i3"
    assert result.method == "bm25"
    assert result.metadata["query_source"] == "history_item_text_only"


def test_bm25_retriever_returns_top_k() -> None:
    retriever = BM25Retriever(text_policy="title")
    retriever.fit([], _catalog())
    result = retriever.retrieve(
        {"example_id": "u:1", "user_id": "u", "history": ["i1"], "target": "i2", "split": "test", "domain": "tiny"},
        k=2,
    )
    assert result.items[0] == "i1" or result.items[0] == "i3"
    assert len(result.items) == 2

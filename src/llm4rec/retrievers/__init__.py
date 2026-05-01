"""Candidate retrievers for Phase 2 baselines."""

from __future__ import annotations

from llm4rec.retrievers.base import BaseRetriever, RetrievalResult
from llm4rec.retrievers.bm25 import BM25Retriever
from llm4rec.retrievers.popularity import PopularityRetriever

__all__ = ["BaseRetriever", "RetrievalResult", "BM25Retriever", "PopularityRetriever"]

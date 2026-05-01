"""Baseline rankers for the unified LLM4Rec prediction schema."""

from __future__ import annotations

from llm4rec.rankers.base import BaseRanker, RankingResult
from llm4rec.rankers.bm25 import BM25Ranker
from llm4rec.rankers.llm_generative import LLMConfidenceObservationRanker, LLMGenerativeRanker
from llm4rec.rankers.llm_reranker import LLMReranker
from llm4rec.rankers.mf import MatrixFactorizationRanker
from llm4rec.rankers.popularity import PopularityRanker
from llm4rec.rankers.random import RandomRanker
from llm4rec.rankers.sequential import MarkovSequentialRanker, SasrecInterfaceRanker, SequentialLastItemRanker

__all__ = [
    "BaseRanker",
    "RankingResult",
    "BM25Ranker",
    "LLMConfidenceObservationRanker",
    "LLMGenerativeRanker",
    "LLMReranker",
    "MatrixFactorizationRanker",
    "PopularityRanker",
    "RandomRanker",
    "MarkovSequentialRanker",
    "SasrecInterfaceRanker",
    "SequentialLastItemRanker",
]

"""Dataset interfaces and tiny preprocessing for the LLM4Rec scaffold."""

from __future__ import annotations

from llm4rec.data.base import Interaction, ItemRecord, UserExample
from llm4rec.data.preprocess import preprocess_dataset

__all__ = [
    "Interaction",
    "ItemRecord",
    "UserExample",
    "preprocess_dataset",
]

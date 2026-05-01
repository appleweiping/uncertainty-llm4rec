"""Trainer interfaces for future baseline training."""

from __future__ import annotations

from llm4rec.trainers.base import BaseTrainer, TrainResult
from llm4rec.trainers.traditional import TraditionalRankerTrainer

__all__ = ["BaseTrainer", "TrainResult", "TraditionalRankerTrainer"]

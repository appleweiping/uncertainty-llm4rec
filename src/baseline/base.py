from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaselineAdapter(ABC):
    def __init__(self, baseline_name: str):
        self.baseline_name = baseline_name

    def fit(self, train_data: list[dict[str, Any]], valid_data: list[dict[str, Any]] | None = None) -> None:
        return None

    @abstractmethod
    def predict_group(self, grouped_sample: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def predict_groups(self, grouped_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.predict_group(sample) for sample in grouped_samples]

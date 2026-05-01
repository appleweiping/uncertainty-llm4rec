"""Trainer contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class TrainResult:
    method: str
    artifact_dir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    checkpoint_dir: str | None = None


class BaseTrainer(Protocol):
    def train(self) -> TrainResult:
        """Train and return a manifest-like result."""

    def evaluate(self) -> dict[str, Any]:
        """Optional trainer-local evaluation hook."""

    def predict(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return prediction-schema records for examples."""

    def fit_predict(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Train, then return prediction-schema records."""

    def save_checkpoint(self, path: str | Path) -> None:
        """Save trainer checkpoint if supported."""

    def load_checkpoint(self, path: str | Path) -> None:
        """Load trainer checkpoint if supported."""

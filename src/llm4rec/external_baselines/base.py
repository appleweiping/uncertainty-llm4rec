"""Common contracts for external baseline adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExternalBaselineConfig:
    """Resolved baseline configuration for an external training/scoring tool."""

    name: str
    model_name: str
    source_project: str
    dataset_name: str
    processed_dir: Path
    output_dir: Path
    seed: int
    candidate_protocol: dict[str, Any] = field(default_factory=dict)
    training_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalBaselineRun:
    """Files produced or consumed by an external baseline adapter."""

    config: ExternalBaselineConfig
    exported_dir: Path
    checkpoint_path: Path | None = None
    raw_predictions_path: Path | None = None
    truce_predictions_path: Path | None = None
    metrics_path: Path | None = None


class MissingExternalDependencyError(RuntimeError):
    """Raised when an optional external baseline dependency is not installed."""

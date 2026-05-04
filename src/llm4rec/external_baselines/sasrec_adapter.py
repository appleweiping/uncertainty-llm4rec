"""SASRec adapter metadata for RecBole-backed external baseline runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.external_baselines.base import ExternalBaselineConfig


def sasrec_config(*, dataset_name: str, processed_dir: str | Path, output_dir: str | Path, seed: int, training_config: dict[str, Any] | None = None) -> ExternalBaselineConfig:
    return ExternalBaselineConfig(
        name="sasrec_recbole",
        model_name="SASRec",
        source_project="RecBole",
        dataset_name=dataset_name,
        processed_dir=Path(processed_dir),
        output_dir=Path(output_dir),
        seed=int(seed),
        training_config=training_config or {},
    )

"""Dataset registry for config-driven preprocessing."""

from __future__ import annotations

from typing import Any, Callable

from llm4rec.data.base import BaseDataModule

DataModuleFactory = Callable[[dict[str, Any]], BaseDataModule]

_DATASETS: dict[str, DataModuleFactory] = {}


def register_dataset(dataset_type: str, factory: DataModuleFactory) -> None:
    if not dataset_type.strip():
        raise ValueError("dataset_type must be non-empty")
    _DATASETS[dataset_type] = factory


def get_dataset_factory(dataset_type: str) -> DataModuleFactory:
    try:
        return _DATASETS[dataset_type]
    except KeyError as exc:
        known = ", ".join(sorted(_DATASETS)) or "<none>"
        raise ValueError(f"unknown dataset type {dataset_type!r}; known: {known}") from exc


def build_data_module(config: dict[str, Any]) -> BaseDataModule:
    dataset_type = str(config.get("type") or config.get("dataset_type") or "")
    return get_dataset_factory(dataset_type)(config)

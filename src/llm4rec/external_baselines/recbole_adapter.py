"""RecBole adapter contract for paper-grade external baselines."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from llm4rec.external_baselines.base import ExternalBaselineConfig, MissingExternalDependencyError


def ensure_recbole_available() -> None:
    if importlib.util.find_spec("recbole") is None:
        raise MissingExternalDependencyError(
            "RecBole is not installed. Install optional baselines dependencies with "
            "`py -3 -m pip install -e .[baselines]` in an environment compatible with RecBole."
        )


def build_recbole_config(config: ExternalBaselineConfig, *, exported_dataset_dir: str | Path) -> dict[str, Any]:
    """Build a RecBole config dictionary without importing RecBole."""

    dataset_dir = Path(exported_dataset_dir)
    return {
        "model": config.model_name,
        "dataset": config.dataset_name,
        "data_path": str(dataset_dir.parent),
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "eval_args": {"split": {"RS": [8, 1, 1]}, "group_by": "user", "order": "TO", "mode": "full"},
        "train_neg_sample_args": None,
        "epochs": int(config.training_config.get("epochs", 100)),
        "train_batch_size": int(config.training_config.get("train_batch_size", 2048)),
        "eval_batch_size": int(config.training_config.get("eval_batch_size", 4096)),
        "learning_rate": float(config.training_config.get("learning_rate", 0.001)),
        "seed": int(config.seed),
        "reproducibility": True,
        "checkpoint_dir": str(config.output_dir / "checkpoints"),
    }


def write_recbole_config(path: str | Path, config: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return out


def run_recbole_training(_: ExternalBaselineConfig, *, exported_dataset_dir: str | Path) -> None:
    """Fail clearly unless RecBole is installed; training implementation is adapter-gated."""

    ensure_recbole_available()
    raise NotImplementedError(
        "RecBole is available, but automated training/scoring is not wired in this run. "
        "Use the exported atomic files and config, then import candidate scores through "
        "`scripts/import_external_predictions.py`."
    )

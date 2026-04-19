from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import ensure_dir


REGISTRY_COLUMNS = [
    "exp_name",
    "domain",
    "task",
    "model",
    "status",
    "config_path",
    "input_path",
    "output_dir",
    "eval_ready",
    "notes",
]


def write_registry(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df = pd.DataFrame(rows)
    for column in REGISTRY_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    df = df[REGISTRY_COLUMNS]
    df.to_csv(path, index=False)
    return path


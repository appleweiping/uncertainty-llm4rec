from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import ensure_dir


REGISTRY_COLUMNS = [
    "batch_name",
    "exp_name",
    "domain",
    "task",
    "model",
    "method_family",
    "method_variant",
    "is_current_best_family",
    "status",
    "config_path",
    "input_path",
    "output_dir",
    "eval_ready",
    "prediction_ready",
    "retry_count",
    "latency_sec",
    "return_code",
    "started_at",
    "finished_at",
    "command",
    "stdout_path",
    "stderr_path",
    "dry_run",
    "error_message",
    "notes",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_registry(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


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


def failed_exp_names(path: str | Path) -> set[str]:
    rows = read_registry(path)
    failed_status = {"failed", "launch_failed", "input_missing"}
    return {
        str(row.get("exp_name"))
        for row in rows
        if str(row.get("status", "")).strip().lower() in failed_status
    }

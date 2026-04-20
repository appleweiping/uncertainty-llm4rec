from __future__ import annotations

import csv
from pathlib import Path


def append_training_status(status_row: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: list[dict] = []
    fieldnames = list(status_row.keys())
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            if reader.fieldnames:
                fieldnames = list(dict.fromkeys([*reader.fieldnames, *fieldnames]))

    existing_rows.append(status_row)
    deduped: list[dict] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    for row in reversed(existing_rows):
        key = (
            str(row.get("run_name", "")),
            str(row.get("stage", "")),
            str(row.get("status", "")),
            str(row.get("adapter_output_dir", "")),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(row)
    deduped.reverse()

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in deduped:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def save_training_summary(summary_row: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary_row.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary_row)

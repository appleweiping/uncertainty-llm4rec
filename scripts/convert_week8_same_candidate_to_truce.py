#!/usr/bin/env python3
"""Convert Week8 same-candidate tasks into TRUCE processed artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--split", choices=["valid", "test"], required=True)
    args = parser.parse_args()
    manifest = convert(task_dir=args.task_dir, output_dir=args.output_dir, domain=args.domain, split=args.split)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def convert(*, task_dir: Path, output_dir: Path, domain: str, split: str) -> dict[str, Any]:
    if not task_dir.exists():
        raise SystemExit(f"task_dir not found: {task_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = task_dir / f"ranking_{split}.jsonl"
    if not ranking_path.exists():
        raise SystemExit(f"ranking file not found: {ranking_path}")
    metadata_path = task_dir / "item_metadata.csv"
    interactions_path = task_dir / "train_interactions.csv"
    metadata_rows = _read_csv_by_item(metadata_path) if metadata_path.exists() else {}
    interactions = _read_interactions(interactions_path) if interactions_path.exists() else {}
    examples = []
    candidate_rows = []
    item_ids: set[str] = set()
    for index, row in enumerate(_read_jsonl(ranking_path), start=1):
        event_id = _first(row, ["event_id", "source_event_id", "example_id"], f"{domain}_{split}_{index}")
        user_id = str(_first(row, ["user_id", "uid"], ""))
        target = str(_first(row, ["target_item", "target_item_id", "positive_item_id", "item_id", "pos_item"], ""))
        candidates = _candidate_items(row)
        if target and target not in candidates:
            candidates = [target] + candidates
        history = _history(row) or interactions.get(user_id, [])
        example = {
            "example_id": str(event_id),
            "user_id": user_id,
            "history": [str(item) for item in history],
            "target": target,
            "candidates": [str(item) for item in candidates],
            "split": split,
            "domain": domain,
            "metadata": {
                "event_id": str(event_id),
                "source_event_id": str(_first(row, ["source_event_id"], event_id)),
                "week8_task_dir": str(task_dir),
                "same_candidate_protocol": True,
                "negative_sampling": "popularity",
                "test_history_mode": "train_plus_valid",
            },
        }
        examples.append(example)
        for item_id in example["candidates"]:
            item_ids.add(item_id)
            candidate_rows.append({
                "example_id": example["example_id"],
                "user_id": user_id,
                "item_id": item_id,
                "split": split,
                "label": int(item_id == target),
            })
        for item_id in example["history"]:
            item_ids.add(item_id)
        if target:
            item_ids.add(target)
    _write_jsonl(output_dir / "examples.jsonl", examples)
    _write_jsonl(output_dir / "candidate_sets.jsonl", candidate_rows)
    _write_items(output_dir / "items.csv", item_ids=item_ids, metadata_rows=metadata_rows, domain=domain)
    _write_interactions(output_dir / "interactions.csv", interactions=interactions, domain=domain)
    manifest = {
        "schema_version": "truce_week8_same_candidate_processed_v1",
        "source_task_dir": str(task_dir),
        "output_dir": str(output_dir),
        "domain": domain,
        "split": split,
        "example_count": len(examples),
        "candidate_row_count": len(candidate_rows),
        "item_count": len(item_ids),
        "preserve_event_alignment": True,
        "do_not_resample": True,
    }
    (output_dir / "preprocess_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_csv_by_item(path: Path) -> dict[str, dict[str, str]]:
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            item_id = str(_first(row, ["item_id", "asin", "iid"], ""))
            if item_id:
                rows[item_id] = {str(k): str(v or "") for k, v in row.items()}
    return rows


def _read_interactions(path: Path) -> dict[str, list[str]]:
    histories: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            user_id = str(_first(row, ["user_id", "uid"], ""))
            item_id = str(_first(row, ["item_id", "asin", "iid"], ""))
            if user_id and item_id:
                histories.setdefault(user_id, []).append(item_id)
    return histories


def _candidate_items(row: dict[str, Any]) -> list[str]:
    for key in ["candidate_items", "candidates", "candidate_item_ids", "items"]:
        value = row.get(key)
        if isinstance(value, list):
            return [str(_candidate_item_id(item)) for item in value]
    return []


def _candidate_item_id(value: Any) -> Any:
    if isinstance(value, dict):
        return _first(value, ["item_id", "asin", "iid"], "")
    return value


def _history(row: dict[str, Any]) -> list[str]:
    for key in ["history", "history_item_ids", "user_history", "train_history"]:
        value = row.get(key)
        if isinstance(value, list):
            return [str(_candidate_item_id(item)) for item in value]
    return []


def _first(row: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_items(path: Path, *, item_ids: set[str], metadata_rows: dict[str, dict[str, str]], domain: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["item_id", "title", "description", "category", "brand", "domain", "raw_text"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item_id in sorted(item_ids):
            meta = metadata_rows.get(item_id, {})
            title = str(_first(meta, ["title", "name", "item_title"], item_id))
            category = str(_first(meta, ["category", "categories"], ""))
            description = str(_first(meta, ["description", "desc"], ""))
            brand = str(_first(meta, ["brand"], ""))
            raw_text = str(_first(meta, ["raw_text", "text"], title))
            writer.writerow({
                "item_id": item_id,
                "title": title,
                "description": description,
                "category": category,
                "brand": brand,
                "domain": domain,
                "raw_text": raw_text,
            })


def _write_interactions(path: Path, *, interactions: dict[str, list[str]], domain: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user_id", "item_id", "timestamp", "rating", "domain"])
        writer.writeheader()
        for user_id, items in sorted(interactions.items()):
            for timestamp, item_id in enumerate(items, start=1):
                writer.writerow({
                    "user_id": user_id,
                    "item_id": item_id,
                    "timestamp": timestamp,
                    "rating": "",
                    "domain": domain,
                })


if __name__ == "__main__":
    raise SystemExit(main())

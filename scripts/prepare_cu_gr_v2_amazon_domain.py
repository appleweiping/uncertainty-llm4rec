#!/usr/bin/env python3
"""Convert prepared Amazon observation data into R3-style CU-GR v2 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--candidate-size", type=int, default=500)
    parser.add_argument("--candidate-size-requested", type=int, default=None)
    parser.add_argument("--candidate-seed", type=int, default=13)
    args = parser.parse_args()

    source = args.source_dir
    output = args.output_dir
    items = _read_csv(source / "item_catalog.csv")
    interactions = _read_csv(source / "interactions.csv")
    examples = _read_jsonl(source / "observation_examples.jsonl")
    popularity = {str(row["item_id"]): int(row.get("popularity") or 0) for row in _read_csv(source / "item_popularity.csv")}

    item_ids = sorted(str(row["item_id"]) for row in items)
    item_title = {str(row["item_id"]): str(row.get("title") or row["item_id"]) for row in items}
    item_bucket = {
        str(row["item_id"]): str(row.get("popularity_bucket") or _bucket_from_popularity(int(popularity.get(str(row["item_id"]), 0))))
        for row in items
    }

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output / "items.csv",
        [
            {
                "item_id": str(row["item_id"]),
                "title": str(row.get("title") or row["item_id"]),
                "description": "",
                "category": args.domain,
                "brand": "",
                "domain": args.domain,
                "raw_text": str(row.get("title") or row["item_id"]),
            }
            for row in items
        ],
    )
    _write_csv(
        output / "interactions.csv",
        [
            {
                "user_id": str(row["user_id"]),
                "item_id": str(row["item_id"]),
                "timestamp": row.get("timestamp", ""),
                "rating": row.get("rating", ""),
                "domain": args.domain,
            }
            for row in interactions
        ],
    )

    out_examples: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for row in examples:
        target = str(row["target_item_id"])
        history = [str(x) for x in row.get("history_item_ids") or []]
        history_titles = [str(x) for x in row.get("history_item_titles") or []]
        candidates = _candidate_items(
            item_ids=item_ids,
            target=target,
            history=history,
            sample_size=args.candidate_size,
            seed=args.candidate_seed,
            example_id=str(row["example_id"]),
        )
        out = {
            "example_id": str(row["example_id"]),
            "user_id": str(row["user_id"]),
            "history": history,
            "history_titles": history_titles,
            "history_item_ids": history,
            "target": target,
            "target_title": str(row.get("target_title") or item_title.get(target, "")),
            "candidates": candidates,
            "split": str(row.get("split") or ""),
            "domain": args.domain,
            "metadata": {
                "example_id": str(row["example_id"]),
                "history_titles": history_titles,
                "history_item_ids": history,
                "target_title": str(row.get("target_title") or item_title.get(target, "")),
                "target_popularity_bucket": str(row.get("target_popularity_bucket") or item_bucket.get(target, "unknown")),
                "source_observation_example": True,
            },
        }
        out_examples.append(out)
        candidate_rows.append(
            {
                "example_id": out["example_id"],
                "user_id": out["user_id"],
                "target_item": target,
                "candidate_items": candidates,
                "split": out["split"],
                "domain": args.domain,
            }
        )

    _write_jsonl(output / "examples.jsonl", out_examples)
    _write_jsonl(output / "candidate_sets.jsonl", candidate_rows)

    split_counts: dict[str, int] = {}
    for row in out_examples:
        split = str(row["split"])
        split_counts[split] = split_counts.get(split, 0) + 1
    manifest = {
        "dataset": args.dataset_name,
        "domain": args.domain,
        "source_dir": str(source),
        "processed_dir": str(output),
        "candidate_protocol": "sampled",
        "candidate_size_requested": int(args.candidate_size_requested or args.candidate_size),
        "candidate_size_effective_max": max((len(row["candidate_items"]) for row in candidate_rows), default=0),
        "candidate_seed": args.candidate_seed,
        "item_count": len(items),
        "interaction_count": len(interactions),
        "example_count": len(out_examples),
        "split_counts": split_counts,
        "leakage_note": "Examples preserve prepared Amazon chronological splits; candidates exclude history items except the held-out target.",
        "outputs": {
            "items": str(output / "items.csv"),
            "interactions": str(output / "interactions.csv"),
            "examples": str(output / "examples.jsonl"),
            "candidate_sets": str(output / "candidate_sets.jsonl"),
        },
        "is_experiment_result": False,
    }
    (output / "preprocess_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def _candidate_items(*, item_ids: list[str], target: str, history: list[str], sample_size: int, seed: int, example_id: str) -> list[str]:
    history_set = set(history)
    pool = [item for item in item_ids if item not in history_set or item == target]
    if target not in pool:
        pool.append(target)
    negatives = sorted({item for item in pool if item != target})
    rng = random.Random(f"{seed}:{example_id}")
    rng.shuffle(negatives)
    selected = [target] + negatives[: max(0, sample_size - 1)]
    return sorted(set(selected))


def _bucket_from_popularity(pop: int) -> str:
    if pop >= 10:
        return "head"
    if pop <= 2:
        return "tail"
    return "mid"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())

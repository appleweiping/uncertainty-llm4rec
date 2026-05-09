#!/usr/bin/env python3
"""Build Qwen observation inputs from Week8 same-candidate processed data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.generation import build_prompt, compute_prompt_hash  # noqa: E402
from storyflow.metrics import assign_popularity_buckets  # noqa: E402
from storyflow.observation import write_jsonl  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--split", choices=["valid", "test"], required=True)
    parser.add_argument("--prompt-template", default="forced_json")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--catalog-csv", type=Path)
    args = parser.parse_args(argv)

    manifest = build_week8_observation_inputs(
        processed_dir=args.processed_dir,
        dataset=args.dataset,
        domain=args.domain,
        split=args.split,
        prompt_template=args.prompt_template,
        max_examples=args.max_examples,
        output_jsonl=args.output_jsonl,
        catalog_csv=args.catalog_csv,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def build_week8_observation_inputs(
    *,
    processed_dir: Path,
    dataset: str,
    domain: str,
    split: str,
    prompt_template: str = "forced_json",
    max_examples: int | None = None,
    output_jsonl: Path | None = None,
    catalog_csv: Path | None = None,
) -> dict[str, Any]:
    """Create API/Qwen observation-compatible prompt inputs for Week8 tasks."""

    processed_dir = processed_dir.resolve()
    if not processed_dir.exists():
        raise SystemExit(f"processed_dir not found: {processed_dir}")
    examples_path = processed_dir / "examples.jsonl"
    items_path = processed_dir / "items.csv"
    interactions_path = processed_dir / "interactions.csv"
    for path in [examples_path, items_path, interactions_path]:
        if not path.exists():
            raise SystemExit(f"required processed file not found: {path}")

    examples = [row for row in _read_jsonl(examples_path) if str(row.get("split")) == split]
    if max_examples is not None:
        if max_examples < 1:
            raise ValueError("max_examples must be >= 1")
        examples = examples[:max_examples]
    if not examples:
        raise ValueError(f"no examples found for split={split} in {examples_path}")

    items = _read_items(items_path)
    popularity = _read_popularity(interactions_path)
    catalog_rows = _build_catalog_rows(items, popularity, domain=domain)

    catalog_path = (
        catalog_csv.resolve()
        if catalog_csv is not None
        else processed_dir / "observation_item_catalog.csv"
    )
    _write_catalog(catalog_path, catalog_rows)
    catalog_by_id = {row["item_id"]: row for row in catalog_rows}

    output_path = (
        output_jsonl.resolve()
        if output_jsonl is not None
        else _default_output_jsonl(dataset=dataset, split=split, prompt_template=prompt_template)
    )
    records = [
        _build_record(
            example,
            dataset=dataset,
            domain=domain,
            split=split,
            processed_dir=processed_dir,
            catalog_csv=catalog_path,
            examples_path=examples_path,
            catalog_by_id=catalog_by_id,
            prompt_template=prompt_template,
        )
        for example in examples
    ]
    write_jsonl(output_path, records)

    bucket_counts = Counter(str(row["target_popularity_bucket"]) for row in records)
    manifest = {
        "schema_version": "truce_week8_observation_inputs_v1",
        "dataset": dataset,
        "domain": domain,
        "split": split,
        "prompt_template": prompt_template,
        "processed_dir": str(processed_dir),
        "examples_jsonl": str(examples_path),
        "source_items_csv": str(items_path),
        "source_interactions_csv": str(interactions_path),
        "catalog_csv": str(catalog_path),
        "output_jsonl": str(output_path),
        "input_count": len(records),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "max_examples": max_examples,
        "is_experiment_result": False,
        "is_paper_result": False,
        "note": (
            "Observation inputs only. Run scripts/server/run_qwen3_observation.py "
            "on approved server hardware for model outputs."
        ),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _default_output_jsonl(*, dataset: str, split: str, prompt_template: str) -> Path:
    return (
        ROOT
        / "outputs"
        / "observation_inputs"
        / "week8_same_candidate"
        / dataset
        / f"{split}_{prompt_template}.jsonl"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_items(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row["item_id"]): {str(key): str(value or "") for key, value in row.items()}
            for row in csv.DictReader(handle)
            if str(row.get("item_id") or "").strip()
        }


def _read_popularity(path: Path) -> dict[str, int]:
    popularity: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            item_id = str(row.get("item_id") or "").strip()
            if item_id:
                popularity[item_id] += 1
    return dict(popularity)


def _build_catalog_rows(
    items: dict[str, dict[str, str]],
    popularity: dict[str, int],
    *,
    domain: str,
) -> list[dict[str, Any]]:
    popularity_for_buckets = {item_id: float(popularity.get(item_id, 0)) for item_id in items}
    if not popularity_for_buckets:
        raise ValueError("items.csv contains no item_id rows")
    buckets = assign_popularity_buckets(popularity_for_buckets)
    rows = []
    for item_id in sorted(items):
        item = items[item_id]
        title = str(item.get("title") or item_id)
        rows.append(
            {
                "item_id": item_id,
                "title": title,
                "title_normalized": _normalize_title(title),
                "popularity": int(popularity.get(item_id, 0)),
                "popularity_bucket": buckets[item_id].value,
                "description": str(item.get("description") or ""),
                "category": str(item.get("category") or ""),
                "brand": str(item.get("brand") or ""),
                "domain": str(item.get("domain") or domain),
                "raw_text": str(item.get("raw_text") or title),
            }
        )
    return rows


def _write_catalog(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id",
        "title",
        "title_normalized",
        "popularity",
        "popularity_bucket",
        "description",
        "category",
        "brand",
        "domain",
        "raw_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_record(
    example: dict[str, Any],
    *,
    dataset: str,
    domain: str,
    split: str,
    processed_dir: Path,
    catalog_csv: Path,
    examples_path: Path,
    catalog_by_id: dict[str, dict[str, Any]],
    prompt_template: str,
) -> dict[str, Any]:
    history_ids = [str(item_id) for item_id in example.get("history", [])]
    target_item_id = str(example.get("target") or "")
    if not target_item_id:
        raise ValueError(f"example {example.get('example_id')} is missing target")
    missing_history = [item_id for item_id in history_ids if item_id not in catalog_by_id]
    if missing_history:
        raise ValueError(
            f"example {example.get('example_id')} has history ids missing from catalog: "
            f"{missing_history[:5]}"
        )
    if target_item_id not in catalog_by_id:
        raise ValueError(
            f"example {example.get('example_id')} target missing from catalog: {target_item_id}"
        )

    history_titles = [str(catalog_by_id[item_id]["title"]) for item_id in history_ids]
    if not history_titles:
        raise ValueError(f"example {example.get('example_id')} has empty history")
    target = catalog_by_id[target_item_id]
    prompt = build_prompt(history_titles, template=prompt_template)
    prompt_hash = compute_prompt_hash(prompt)
    example_id = str(example.get("example_id"))
    metadata = dict(example.get("metadata") or {})
    return {
        "input_id": f"week8:{dataset}:{split}:{example_id}:{prompt_hash[:12]}",
        "dataset": dataset,
        "domain": domain,
        "processed_suffix": split,
        "example_id": example_id,
        "event_id": str(metadata.get("event_id") or example_id),
        "source_event_id": str(metadata.get("source_event_id") or metadata.get("event_id") or example_id),
        "user_id": str(example.get("user_id")),
        "split": split,
        "history_item_ids": history_ids,
        "history_item_titles": history_titles,
        "history_timestamps": list(range(1, len(history_ids) + 1)),
        "history_length": len(history_ids),
        "target_in_history": target_item_id in history_ids,
        "target_history_occurrence_count": history_ids.count(target_item_id),
        "target_same_timestamp_as_history": False,
        "history_duplicate_item_count": _duplicate_count(history_ids),
        "history_unique_item_count": len(set(history_ids)),
        "target_item_id": target_item_id,
        "target_title": str(target["title"]),
        "target_timestamp": len(history_ids) + 1,
        "target_popularity": int(target["popularity"]),
        "target_popularity_bucket": str(target["popularity_bucket"]),
        "prompt_template": prompt_template,
        "repeat_target_policy": "all",
        "prompt": prompt,
        "prompt_hash": prompt_hash,
        "source": {
            "processed_dir": str(processed_dir),
            "catalog_csv": str(catalog_csv),
            "observation_examples": str(examples_path),
            "week8_same_candidate": True,
            "same_candidate_protocol": True,
        },
    }


def _duplicate_count(values: list[str]) -> int:
    counts = Counter(values)
    return sum(count - 1 for count in counts.values() if count > 1)


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().split())


if __name__ == "__main__":
    raise SystemExit(main())

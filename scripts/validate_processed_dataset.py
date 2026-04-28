"""Validate processed Storyflow dataset outputs before observation runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "item_catalog.csv",
    "interactions.csv",
    "item_popularity.csv",
    "user_sequences.jsonl",
    "observation_examples.jsonl",
    "preprocess_manifest.json",
]

REQUIRED_EXAMPLE_FIELDS = {
    "example_id",
    "user_id",
    "history_item_ids",
    "history_item_titles",
    "target_item_id",
    "target_item_title",
    "target_title",
    "split",
    "target_timestamp",
    "item_popularity",
    "target_item_popularity",
    "popularity_bucket",
    "target_popularity_bucket",
}

REQUIRED_MANIFEST_FIELDS = {
    "dataset",
    "generated_at_utc",
    "split_policy",
    "min_user_interactions",
    "user_k_core",
    "item_k_core",
    "min_history",
    "max_history",
    "item_count",
    "interaction_count",
    "user_count",
    "example_count",
    "split_counts",
    "outputs",
    "config_snapshot",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _processed_dir(dataset: str, suffix: str) -> Path:
    return ROOT / "data" / "processed" / dataset / suffix


def _add_issue(issues: list[str], message: str) -> None:
    issues.append(message)


def _validate_files(processed_dir: Path, blockers: list[str]) -> dict[str, bool]:
    exists = {}
    if not processed_dir.exists():
        _add_issue(blockers, f"Processed directory does not exist: {processed_dir}")
        return {name: False for name in REQUIRED_FILES}
    for name in REQUIRED_FILES:
        path = processed_dir / name
        exists[name] = path.exists()
        if not path.exists():
            _add_issue(blockers, f"Missing required processed file: {path}")
    return exists


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    blockers: list[str],
) -> None:
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    if missing:
        _add_issue(blockers, f"Manifest missing fields: {missing}")
    if manifest.get("split_policy") not in {"leave_last_one", "leave_last_two", "global_chronological"}:
        _add_issue(blockers, f"Unknown split_policy in manifest: {manifest.get('split_policy')}")


def _validate_examples(
    examples: list[dict[str, Any]],
    *,
    blockers: list[str],
    warnings: list[str],
) -> None:
    if not examples:
        _add_issue(blockers, "observation_examples.jsonl is empty")
        return
    missing_fields = sorted(
        field
        for field in REQUIRED_EXAMPLE_FIELDS
        if any(field not in example for example in examples[:100])
    )
    if missing_fields:
        _add_issue(blockers, f"Observation examples missing required fields: {missing_fields}")

    for index, example in enumerate(examples):
        history_ids = example.get("history_item_ids") or []
        history_titles = example.get("history_item_titles") or []
        if len(history_ids) != len(history_titles):
            _add_issue(
                blockers,
                f"Example {example.get('example_id', index)} has mismatched history ids/titles",
            )
            break
        if not history_ids:
            _add_issue(blockers, f"Example {example.get('example_id', index)} has empty history")
            break
        if not str(example.get("target_title") or example.get("target_item_title") or "").strip():
            _add_issue(blockers, f"Example {example.get('example_id', index)} has empty target title")
            break
        if example.get("target_item_id") in history_ids:
            _add_issue(
                warnings,
                f"Example {example.get('example_id', index)} target appears in history; repeated items may be legitimate but should be inspected.",
            )
            break


def _validate_sequence_alignment(
    sequences: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    *,
    split_policy: str,
    blockers: list[str],
) -> None:
    sequence_by_user = {sequence["user_id"]: sequence for sequence in sequences}
    split_counts: Counter[str] = Counter()
    per_user_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        split_counts[str(example["split"])] += 1
        per_user_examples[str(example["user_id"])].append(example)
        sequence = sequence_by_user.get(str(example["user_id"]))
        if sequence is None:
            _add_issue(blockers, f"Example user missing from user_sequences: {example['user_id']}")
            return
        target_index = int(example.get("target_index", str(example["example_id"]).split(":")[-1]))
        history_start = int(example.get("history_start_index", 0))
        expected_history = sequence["item_ids"][history_start:target_index]
        expected_timestamps = sequence["timestamps"][history_start:target_index]
        if list(example["history_item_ids"]) != expected_history:
            _add_issue(blockers, f"History is not prefix-aligned for {example['example_id']}")
            return
        if example.get("history_timestamps") and list(example["history_timestamps"]) != expected_timestamps:
            _add_issue(blockers, f"History timestamps are not prefix-aligned for {example['example_id']}")
            return
        if str(example["target_item_id"]) != str(sequence["item_ids"][target_index]):
            _add_issue(blockers, f"Target item is not sequence target for {example['example_id']}")
            return
        if int(example["target_timestamp"]) != int(sequence["timestamps"][target_index]):
            _add_issue(blockers, f"Target timestamp is not sequence target for {example['example_id']}")
            return
        if expected_timestamps and max(expected_timestamps) > int(example["target_timestamp"]):
            _add_issue(blockers, f"History contains future timestamp for {example['example_id']}")
            return

    if split_policy == "leave_last_two":
        for user_id, rows in per_user_examples.items():
            sequence = sequence_by_user[user_id]
            val_rows = [row for row in rows if row["split"] == "val"]
            test_rows = [row for row in rows if row["split"] == "test"]
            if len(val_rows) != 1 or len(test_rows) != 1:
                _add_issue(blockers, f"leave_last_two requires one val and one test for user {user_id}")
                return
            if int(val_rows[0]["target_index"]) != len(sequence["item_ids"]) - 2:
                _add_issue(blockers, f"Val target is not second-to-last for user {user_id}")
                return
            if int(test_rows[0]["target_index"]) != len(sequence["item_ids"]) - 1:
                _add_issue(blockers, f"Test target is not last for user {user_id}")
                return


def _validate_titles_and_buckets(
    catalog: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    *,
    blockers: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    empty_titles = [row["item_id"] for row in catalog if not str(row.get("title", "")).strip()]
    if empty_titles:
        _add_issue(blockers, f"Catalog contains empty titles: {empty_titles[:5]}")
    normalized_counts = Counter(row.get("title_normalized", "") for row in catalog)
    duplicates = {
        title: count
        for title, count in normalized_counts.items()
        if title and count > 1
    }
    if duplicates:
        _add_issue(
            warnings,
            f"Catalog has duplicate normalized titles; grounding ambiguity possible: {list(duplicates)[:5]}",
        )
    bucket_counts = Counter(row.get("target_popularity_bucket") for row in examples)
    for bucket in ("head", "mid", "tail"):
        if bucket_counts[bucket] == 0:
            _add_issue(blockers, f"Target popularity bucket is empty in examples: {bucket}")
    return {
        "empty_catalog_title_count": len(empty_titles),
        "duplicate_normalized_title_count": len(duplicates),
        "target_bucket_counts": dict(bucket_counts),
    }


def _write_report(summary: dict[str, Any], report_path: Path) -> None:
    lines = [
        f"# Processed Dataset Validation: {summary['dataset']} / {summary['processed_suffix']}",
        "",
        "This report validates local processed data for observation readiness. It is not an experiment result.",
        "",
        "## Status",
        "",
        f"- Status: {summary['status']}",
        f"- Blockers: {len(summary['blockers'])}",
        f"- Warnings: {len(summary['warnings'])}",
        f"- Output directory: `{summary['processed_dir']}`",
        "",
        "## Counts",
        "",
        f"- Catalog items: {summary['counts'].get('catalog_items')}",
        f"- Interactions: {summary['counts'].get('interactions')}",
        f"- Users: {summary['counts'].get('user_sequences')}",
        f"- Observation examples: {summary['counts'].get('observation_examples')}",
        f"- Split counts: {summary['counts'].get('split_counts')}",
        f"- Target bucket counts: {summary['title_bucket_checks'].get('target_bucket_counts')}",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- {issue}" for issue in summary["blockers"] or ["None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {issue}" for issue in summary["warnings"] or ["None"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_processed_dataset(
    *,
    dataset: str,
    processed_suffix: str,
    output_dir: Path,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    processed_dir = _processed_dir(dataset, processed_suffix)
    files = _validate_files(processed_dir, blockers)
    manifest: dict[str, Any] = {}
    catalog: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    title_bucket_checks: dict[str, Any] = {}

    if all(files.values()):
        manifest = json.loads((processed_dir / "preprocess_manifest.json").read_text(encoding="utf-8"))
        catalog = _read_csv(processed_dir / "item_catalog.csv")
        interactions = _read_csv(processed_dir / "interactions.csv")
        sequences = _read_jsonl(processed_dir / "user_sequences.jsonl")
        examples = _read_jsonl(processed_dir / "observation_examples.jsonl")
        _validate_manifest(manifest, blockers=blockers)
        _validate_examples(examples, blockers=blockers, warnings=warnings)
        _validate_sequence_alignment(
            sequences,
            examples,
            split_policy=str(manifest.get("split_policy", "")),
            blockers=blockers,
        )
        title_bucket_checks = _validate_titles_and_buckets(
            catalog,
            examples,
            blockers=blockers,
            warnings=warnings,
        )

    split_counts = Counter(example.get("split") for example in examples)
    summary = {
        "dataset": dataset,
        "processed_suffix": processed_suffix,
        "processed_dir": str(processed_dir),
        "status": "blocker" if blockers else "ok",
        "blockers": blockers,
        "warnings": warnings,
        "files": files,
        "counts": {
            "catalog_items": len(catalog),
            "interactions": len(interactions),
            "user_sequences": len(sequences),
            "observation_examples": len(examples),
            "split_counts": dict(split_counts),
        },
        "manifest_split_policy": manifest.get("split_policy"),
        "manifest_generated_at_utc": manifest.get("generated_at_utc"),
        "title_bucket_checks": title_bucket_checks,
        "is_experiment_result": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "validation_summary.json"
    report_path = output_dir / "validation_report.md"
    summary["summary_json"] = str(summary_path)
    summary["markdown_report"] = str(report_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(summary, report_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--processed-suffix", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else (
        ROOT
        / "outputs"
        / "data_validation"
        / args.dataset
        / args.processed_suffix
    )
    summary = validate_processed_dataset(
        dataset=args.dataset,
        processed_suffix=args.processed_suffix,
        output_dir=output_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 1 if summary["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

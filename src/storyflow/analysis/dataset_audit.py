"""Audit processed observation examples before scaling observation runs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


REQUIRED_PROCESSED_FILES = [
    "item_catalog.csv",
    "interactions.csv",
    "item_popularity.csv",
    "user_sequences.jsonl",
    "observation_examples.jsonl",
    "preprocess_manifest.json",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ratio(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _numeric_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
    }


def _bucket_counts_by_split(examples: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in examples:
        split = str(row.get("split", "unknown"))
        bucket = str(row.get("target_popularity_bucket") or row.get("popularity_bucket") or "unknown")
        counts[split][bucket] += 1
    return {split: dict(counter) for split, counter in sorted(counts.items())}


def _split_timestamp_summary(examples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_split: dict[str, list[int]] = defaultdict(list)
    for row in examples:
        by_split[str(row.get("split", "unknown"))].append(_as_int(row.get("target_timestamp")))
    return {split: _numeric_summary(values) for split, values in sorted(by_split.items())}


def _global_chronological_checks(
    examples: list[dict[str, Any]],
    split_policy: str,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    summary = {
        "split_policy": split_policy,
        "checked": split_policy == "global_chronological",
        "split_timestamp_summary": _split_timestamp_summary(examples),
        "boundary_warnings": [],
    }
    if split_policy != "global_chronological":
        return summary, warnings

    order = ["train", "val", "test"]
    ranges = summary["split_timestamp_summary"]
    for left, right in zip(order, order[1:]):
        left_max = ranges.get(left, {}).get("max")
        right_min = ranges.get(right, {}).get("min")
        if left_max is not None and right_min is not None and left_max > right_min:
            message = (
                f"Global chronological split boundary overlap: "
                f"{left}.max={left_max} > {right}.min={right_min}"
            )
            summary["boundary_warnings"].append(message)
            warnings.append(message)
    return summary, warnings


def _example_sequence_alignment(
    examples: list[dict[str, Any]],
    sequences: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    by_user = {str(row.get("user_id")): row for row in sequences}
    target_timestamp_violations: list[dict[str, Any]] = []
    prefix_mismatches: list[dict[str, Any]] = []
    target_mismatches: list[dict[str, Any]] = []

    for row in examples:
        example_id = str(row.get("example_id"))
        user_id = str(row.get("user_id"))
        sequence = by_user.get(user_id)
        if sequence is None:
            blockers.append(f"Example user missing from user_sequences: {example_id}")
            break
        target_index = _as_int(row.get("target_index"), -1)
        history_start = _as_int(row.get("history_start_index"), 0)
        item_ids = list(sequence.get("item_ids") or [])
        timestamps = [_as_int(value) for value in sequence.get("timestamps") or []]
        if target_index < 0 or target_index >= len(item_ids):
            blockers.append(f"Invalid target_index for example: {example_id}")
            break
        expected_history = item_ids[history_start:target_index]
        actual_history = list(row.get("history_item_ids") or [])
        if expected_history != actual_history:
            prefix_mismatches.append(
                {
                    "example_id": example_id,
                    "user_id": user_id,
                    "expected_history_length": len(expected_history),
                    "actual_history_length": len(actual_history),
                }
            )
        if str(row.get("target_item_id")) != str(item_ids[target_index]):
            target_mismatches.append(
                {
                    "example_id": example_id,
                    "user_id": user_id,
                    "expected_target_item_id": item_ids[target_index],
                    "actual_target_item_id": row.get("target_item_id"),
                }
            )
        history_timestamps = [_as_int(value) for value in row.get("history_timestamps") or []]
        target_timestamp = _as_int(row.get("target_timestamp"))
        if history_timestamps and max(history_timestamps) > target_timestamp:
            target_timestamp_violations.append(
                {
                    "example_id": example_id,
                    "user_id": user_id,
                    "max_history_timestamp": max(history_timestamps),
                    "target_timestamp": target_timestamp,
                }
            )
        if timestamps and _as_int(row.get("target_timestamp")) != timestamps[target_index]:
            target_timestamp_violations.append(
                {
                    "example_id": example_id,
                    "user_id": user_id,
                    "sequence_target_timestamp": timestamps[target_index],
                    "target_timestamp": row.get("target_timestamp"),
                }
            )

    if prefix_mismatches:
        blockers.append(f"History prefix mismatches: {len(prefix_mismatches)}")
    if target_mismatches:
        blockers.append(f"Target item mismatches: {len(target_mismatches)}")
    if target_timestamp_violations:
        blockers.append(f"Timestamp leakage or target timestamp mismatches: {len(target_timestamp_violations)}")
    if not blockers and not warnings:
        warnings = []

    return (
        {
            "checked_examples": len(examples),
            "prefix_mismatch_count": len(prefix_mismatches),
            "target_mismatch_count": len(target_mismatches),
            "timestamp_violation_count": len(target_timestamp_violations),
            "prefix_mismatch_examples": prefix_mismatches[:20],
            "target_mismatch_examples": target_mismatches[:20],
            "timestamp_violation_examples": target_timestamp_violations[:20],
        },
        blockers,
        warnings,
    )


def _repeat_and_history_checks(examples: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    split_counts: Counter[str] = Counter(str(row.get("split", "unknown")) for row in examples)
    target_in_history_by_split: Counter[str] = Counter()
    duplicate_history_by_split: Counter[str] = Counter()
    same_timestamp_by_split: Counter[str] = Counter()
    repeated_cases: list[dict[str, Any]] = []
    duplicate_history_cases: list[dict[str, Any]] = []

    for row in examples:
        split = str(row.get("split", "unknown"))
        history_ids = list(row.get("history_item_ids") or [])
        target_item_id = row.get("target_item_id")
        duplicate_items = sorted(item_id for item_id, count in Counter(history_ids).items() if count > 1)
        if duplicate_items:
            duplicate_history_by_split[split] += 1
            if len(duplicate_history_cases) < 50:
                duplicate_history_cases.append(
                    {
                        "example_id": row.get("example_id"),
                        "user_id": row.get("user_id"),
                        "split": split,
                        "history_length": len(history_ids),
                        "duplicate_item_ids": duplicate_items[:10],
                        "target_item_id": target_item_id,
                        "target_title": row.get("target_title") or row.get("target_item_title"),
                    }
                )
        if target_item_id in history_ids:
            target_in_history_by_split[split] += 1
            history_timestamps = [_as_int(value) for value in row.get("history_timestamps") or []]
            target_timestamp = _as_int(row.get("target_timestamp"))
            same_timestamp = bool(history_timestamps and target_timestamp in history_timestamps)
            if same_timestamp:
                same_timestamp_by_split[split] += 1
            if len(repeated_cases) < 50:
                repeated_cases.append(
                    {
                        "example_id": row.get("example_id"),
                        "user_id": row.get("user_id"),
                        "split": split,
                        "history_length": len(history_ids),
                        "target_item_id": target_item_id,
                        "target_title": row.get("target_title") or row.get("target_item_title"),
                        "target_timestamp": row.get("target_timestamp"),
                        "history_occurrence_count": history_ids.count(target_item_id),
                        "same_timestamp_as_history": same_timestamp,
                        "target_popularity_bucket": row.get("target_popularity_bucket")
                        or row.get("popularity_bucket"),
                    }
                )

    total_repeated = sum(target_in_history_by_split.values())
    total_duplicates = sum(duplicate_history_by_split.values())
    if total_repeated:
        warnings.append(
            "Target item appears in history for some examples; this can be legitimate repeat "
            "purchase behavior or duplicate review artifacts and should be stratified before paper claims."
        )
    if total_duplicates:
        warnings.append(
            "Some histories contain duplicate item ids; repeat-heavy users may let a model repeat a seen title."
        )

    return (
        {
            "target_in_history_count": total_repeated,
            "target_in_history_rate": _ratio(total_repeated, len(examples)),
            "target_in_history_by_split": dict(target_in_history_by_split),
            "target_in_history_rate_by_split": {
                split: _ratio(target_in_history_by_split[split], split_counts[split])
                for split in sorted(split_counts)
            },
            "same_timestamp_target_in_history_count": sum(same_timestamp_by_split.values()),
            "same_timestamp_target_in_history_by_split": dict(same_timestamp_by_split),
            "duplicate_history_example_count": total_duplicates,
            "duplicate_history_example_rate": _ratio(total_duplicates, len(examples)),
            "duplicate_history_by_split": dict(duplicate_history_by_split),
            "repeated_target_case_samples": repeated_cases,
            "duplicate_history_case_samples": duplicate_history_cases,
        },
        warnings,
    )


def _title_quality_checks(
    catalog: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    empty_catalog_titles = [row.get("item_id") for row in catalog if not str(row.get("title", "")).strip()]
    empty_target_examples = [
        row.get("example_id")
        for row in examples
        if not str(row.get("target_title") or row.get("target_item_title") or "").strip()
    ]
    empty_history_examples = [
        row.get("example_id")
        for row in examples
        if any(not str(title).strip() for title in row.get("history_item_titles") or [])
    ]
    normalized_counts = Counter(str(row.get("title_normalized") or "").strip() for row in catalog)
    duplicate_normalized = {
        title: count for title, count in normalized_counts.items() if title and count > 1
    }
    if empty_catalog_titles:
        blockers.append(f"Empty catalog titles: {len(empty_catalog_titles)}")
    if empty_target_examples:
        blockers.append(f"Empty target titles: {len(empty_target_examples)}")
    if empty_history_examples:
        blockers.append(f"Empty history titles: {len(empty_history_examples)}")
    if duplicate_normalized:
        warnings.append(
            f"Duplicate normalized catalog title groups detected: {len(duplicate_normalized)}"
        )

    return (
        {
            "empty_catalog_title_count": len(empty_catalog_titles),
            "empty_target_title_example_count": len(empty_target_examples),
            "empty_history_title_example_count": len(empty_history_examples),
            "duplicate_normalized_title_group_count": len(duplicate_normalized),
            "duplicate_normalized_title_samples": [
                {"title_normalized": title, "count": count}
                for title, count in list(duplicate_normalized.items())[:20]
            ],
        },
        blockers,
        warnings,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def dataset_audit_markdown(summary: dict[str, Any]) -> str:
    repeat = summary["repeat_history_checks"]
    chronology = summary["chronological_checks"]
    lines = [
        f"# Processed Dataset Audit: {summary['dataset']} / {summary['processed_suffix']}",
        "",
        "This audit is a local data-readiness artifact. It is not an experiment result.",
        "",
        "## Status",
        "",
        f"- Status: {summary['status']}",
        f"- Blockers: {len(summary['blockers'])}",
        f"- Warnings: {len(summary['warnings'])}",
        f"- Processed directory: `{summary['processed_dir']}`",
        "",
        "## Core Counts",
        "",
        f"- Catalog items: {summary['counts']['catalog_items']}",
        f"- Interactions: {summary['counts']['interactions']}",
        f"- Users: {summary['counts']['user_sequences']}",
        f"- Observation examples: {summary['counts']['observation_examples']}",
        f"- Split counts: {summary['counts']['split_counts']}",
        f"- Bucket counts by split: {summary['bucket_counts_by_split']}",
        "",
        "## Repeated-Item And History Checks",
        "",
        f"- Target-in-history examples: {repeat['target_in_history_count']}",
        f"- Target-in-history rate: {repeat['target_in_history_rate']:.4f}",
        f"- Target-in-history by split: {repeat['target_in_history_by_split']}",
        f"- Same-timestamp target-in-history examples: {repeat['same_timestamp_target_in_history_count']}",
        f"- Duplicate-history examples: {repeat['duplicate_history_example_count']}",
        "",
        "## Chronological Checks",
        "",
        f"- Split policy: {chronology['split_policy']}",
        f"- Split timestamp summary: {chronology['split_timestamp_summary']}",
        f"- Boundary warnings: {chronology['boundary_warnings']}",
        f"- Sequence alignment timestamp violations: {summary['sequence_alignment']['timestamp_violation_count']}",
        f"- Sequence alignment prefix mismatches: {summary['sequence_alignment']['prefix_mismatch_count']}",
        "",
        "## Title Quality",
        "",
        f"- Empty catalog titles: {summary['title_quality']['empty_catalog_title_count']}",
        f"- Empty target-title examples: {summary['title_quality']['empty_target_title_example_count']}",
        f"- Empty history-title examples: {summary['title_quality']['empty_history_title_example_count']}",
        f"- Duplicate normalized title groups: {summary['title_quality']['duplicate_normalized_title_group_count']}",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- {issue}" for issue in summary["blockers"] or ["None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {issue}" for issue in summary["warnings"] or ["None"])
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "Repeated items are not automatically data leakage in implicit e-commerce data, "
            "because repeat purchase or duplicate reviews can be real. They are a scale-up risk: "
            "future observation reports should stratify or optionally exclude repeat-target cases "
            "before making paper claims about confidence and correctness.",
            "",
        ]
    )
    return "\n".join(lines)


def audit_processed_dataset(
    *,
    dataset: str,
    processed_suffix: str,
    processed_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    files = {name: (processed_dir / name).exists() for name in REQUIRED_PROCESSED_FILES}
    missing = [name for name, exists in files.items() if not exists]
    if missing:
        blockers.append(f"Missing required processed files: {missing}")

    manifest: dict[str, Any] = {}
    catalog: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    if not blockers:
        manifest = json.loads((processed_dir / "preprocess_manifest.json").read_text(encoding="utf-8"))
        catalog = read_csv(processed_dir / "item_catalog.csv")
        interactions = read_csv(processed_dir / "interactions.csv")
        sequences = read_jsonl(processed_dir / "user_sequences.jsonl")
        examples = read_jsonl(processed_dir / "observation_examples.jsonl")
        if not examples:
            blockers.append("observation_examples.jsonl is empty")

    if examples:
        repeat_checks, repeat_warnings = _repeat_and_history_checks(examples)
        title_quality, title_blockers, title_warnings = _title_quality_checks(catalog, examples)
        sequence_alignment, sequence_blockers, sequence_warnings = _example_sequence_alignment(
            examples,
            sequences,
        )
        chronological_checks, chronological_warnings = _global_chronological_checks(
            examples,
            str(manifest.get("split_policy", "")),
        )
        blockers.extend(title_blockers)
        blockers.extend(sequence_blockers)
        warnings.extend(repeat_warnings)
        warnings.extend(title_warnings)
        warnings.extend(sequence_warnings)
        warnings.extend(chronological_warnings)
    else:
        repeat_checks = {}
        title_quality = {}
        sequence_alignment = {}
        chronological_checks = {
            "split_policy": manifest.get("split_policy"),
            "checked": False,
            "split_timestamp_summary": {},
            "boundary_warnings": [],
        }

    split_counts = Counter(str(row.get("split", "unknown")) for row in examples)
    history_lengths = [_as_int(row.get("history_length"), len(row.get("history_item_ids") or [])) for row in examples]
    summary: dict[str, Any] = {
        "dataset": dataset,
        "processed_suffix": processed_suffix,
        "processed_dir": str(processed_dir),
        "status": "blocker" if blockers else "ok_with_warnings" if warnings else "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "is_experiment_result": False,
        "files": files,
        "blockers": blockers,
        "warnings": warnings,
        "counts": {
            "catalog_items": len(catalog),
            "interactions": len(interactions),
            "user_sequences": len(sequences),
            "observation_examples": len(examples),
            "split_counts": dict(split_counts),
        },
        "manifest": {
            "dataset": manifest.get("dataset"),
            "generated_at_utc": manifest.get("generated_at_utc"),
            "split_policy": manifest.get("split_policy"),
            "is_sample_result": manifest.get("is_sample_result"),
            "is_full_result": manifest.get("is_full_result"),
            "is_experiment_result": manifest.get("is_experiment_result"),
            "min_user_interactions": manifest.get("min_user_interactions"),
            "user_k_core": manifest.get("user_k_core"),
            "item_k_core": manifest.get("item_k_core"),
            "min_history": manifest.get("min_history"),
            "max_history": manifest.get("max_history"),
            "split_counts": manifest.get("split_counts"),
        },
        "history_length_summary": _numeric_summary(history_lengths),
        "bucket_counts_by_split": _bucket_counts_by_split(examples),
        "repeat_history_checks": repeat_checks,
        "title_quality": title_quality,
        "sequence_alignment": sequence_alignment,
        "chronological_checks": chronological_checks,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "dataset_audit_summary.json"
    report_path = output_dir / "dataset_audit_report.md"
    repeated_cases_path = output_dir / "repeated_target_cases.jsonl"
    duplicate_history_path = output_dir / "duplicate_history_cases.jsonl"
    summary["summary_json"] = str(summary_path)
    summary["markdown_report"] = str(report_path)
    summary["repeated_target_cases_jsonl"] = str(repeated_cases_path)
    summary["duplicate_history_cases_jsonl"] = str(duplicate_history_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    report_path.write_text(dataset_audit_markdown(summary) + "\n", encoding="utf-8")
    _write_jsonl(
        repeated_cases_path,
        list((summary.get("repeat_history_checks") or {}).get("repeated_target_case_samples") or []),
    )
    _write_jsonl(
        duplicate_history_path,
        list((summary.get("repeat_history_checks") or {}).get("duplicate_history_case_samples") or []),
    )
    return summary


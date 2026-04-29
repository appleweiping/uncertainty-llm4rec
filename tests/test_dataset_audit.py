from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

from scripts.audit_processed_dataset import main as audit_cli_main
from storyflow.analysis import audit_processed_dataset
from storyflow.analysis.dataset_audit import read_jsonl


def _workspace(name: str) -> Path:
    path = Path("outputs") / "test_tmp" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_processed_fixture(processed_dir: Path, *, future_leak: bool = False) -> None:
    _write_csv(
        processed_dir / "item_catalog.csv",
        ["item_id", "title", "title_normalized", "popularity", "popularity_bucket"],
        [
            {
                "item_id": "a",
                "title": "Alpha Serum",
                "title_normalized": "alpha serum",
                "popularity": 12,
                "popularity_bucket": "head",
            },
            {
                "item_id": "b",
                "title": "Beta Cream",
                "title_normalized": "beta cream",
                "popularity": 6,
                "popularity_bucket": "mid",
            },
            {
                "item_id": "c",
                "title": "Cold Gel",
                "title_normalized": "cold gel",
                "popularity": 1,
                "popularity_bucket": "tail",
            },
        ],
    )
    _write_csv(
        processed_dir / "interactions.csv",
        ["user_id", "item_id", "rating", "timestamp"],
        [
            {"user_id": "u1", "item_id": "a", "rating": 5, "timestamp": 1},
            {"user_id": "u1", "item_id": "b", "rating": 4, "timestamp": 2},
            {"user_id": "u1", "item_id": "a", "rating": 5, "timestamp": 3},
        ],
    )
    _write_csv(
        processed_dir / "item_popularity.csv",
        ["item_id", "popularity", "popularity_bucket"],
        [
            {"item_id": "a", "popularity": 12, "popularity_bucket": "head"},
            {"item_id": "b", "popularity": 6, "popularity_bucket": "mid"},
            {"item_id": "c", "popularity": 1, "popularity_bucket": "tail"},
        ],
    )
    timestamps = [1, 2, 3]
    if future_leak:
        timestamps = [5, 2, 3]
    _write_jsonl(
        processed_dir / "user_sequences.jsonl",
        [
            {
                "user_id": "u1",
                "item_ids": ["a", "b", "a"],
                "ratings": [5, 4, 5],
                "timestamps": timestamps,
            }
        ],
    )
    _write_jsonl(
        processed_dir / "observation_examples.jsonl",
        [
            {
                "example_id": "u1:1",
                "user_id": "u1",
                "history_start_index": 0,
                "history_end_index": 0,
                "target_index": 1,
                "history_item_ids": ["a"],
                "history_item_titles": ["Alpha Serum"],
                "history_timestamps": [timestamps[0]],
                "history_length": 1,
                "target_item_id": "b",
                "target_title": "Beta Cream",
                "target_item_title": "Beta Cream",
                "target_timestamp": timestamps[1],
                "target_item_popularity": 6,
                "target_popularity_bucket": "mid",
                "split": "train",
            },
            {
                "example_id": "u1:2",
                "user_id": "u1",
                "history_start_index": 0,
                "history_end_index": 1,
                "target_index": 2,
                "history_item_ids": ["a", "b"],
                "history_item_titles": ["Alpha Serum", "Beta Cream"],
                "history_timestamps": [timestamps[0], timestamps[1]],
                "history_length": 2,
                "target_item_id": "a",
                "target_title": "Alpha Serum",
                "target_item_title": "Alpha Serum",
                "target_timestamp": timestamps[2],
                "target_item_popularity": 12,
                "target_popularity_bucket": "head",
                "split": "test",
            },
        ],
    )
    (processed_dir / "preprocess_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "fixture",
                "generated_at_utc": "2026-04-29T00:00:00+00:00",
                "split_policy": "global_chronological",
                "is_sample_result": False,
                "is_full_result": True,
                "is_experiment_result": False,
                "min_user_interactions": 1,
                "user_k_core": 1,
                "item_k_core": 1,
                "min_history": 1,
                "max_history": 50,
                "split_counts": {"train": 1, "test": 1},
            }
        ),
        encoding="utf-8",
    )


def test_dataset_audit_counts_repeated_target_without_blocking() -> None:
    workspace = _workspace("dataset_audit_repeat")
    processed = workspace / "processed"
    output = workspace / "out"
    _write_processed_fixture(processed)

    summary = audit_processed_dataset(
        dataset="fixture",
        processed_suffix="tiny",
        processed_dir=processed,
        output_dir=output,
    )

    repeated_cases = read_jsonl(output / "repeated_target_cases.jsonl")
    assert summary["status"] == "ok_with_warnings"
    assert summary["blockers"] == []
    assert summary["repeat_history_checks"]["target_in_history_count"] == 1
    assert summary["repeat_history_checks"]["target_in_history_by_split"] == {"test": 1}
    assert summary["sequence_alignment"]["timestamp_violation_count"] == 0
    assert repeated_cases[0]["example_id"] == "u1:2"
    assert (output / "dataset_audit_report.md").exists()


def test_dataset_audit_blocks_future_history_timestamp() -> None:
    workspace = _workspace("dataset_audit_future")
    processed = workspace / "processed"
    output = workspace / "out"
    _write_processed_fixture(processed, future_leak=True)

    summary = audit_processed_dataset(
        dataset="fixture",
        processed_suffix="tiny",
        processed_dir=processed,
        output_dir=output,
    )

    assert summary["status"] == "blocker"
    assert summary["sequence_alignment"]["timestamp_violation_count"] == 2
    assert any("Timestamp leakage" in issue for issue in summary["blockers"])


def test_dataset_audit_cli_returns_nonzero_for_blocker() -> None:
    workspace = _workspace("dataset_audit_cli")
    processed = workspace / "processed"
    output = workspace / "out"
    _write_processed_fixture(processed, future_leak=True)

    code = audit_cli_main(
        [
            "--dataset",
            "fixture",
            "--processed-suffix",
            "tiny",
            "--processed-dir",
            str(processed),
            "--output-dir",
            str(output),
        ]
    )

    assert code == 1
    summary = json.loads((output / "dataset_audit_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "blocker"


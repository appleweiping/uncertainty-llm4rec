from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

from scripts.run_baseline_observation import main as baseline_main
from storyflow.baselines import (
    CooccurrenceTitleBaseline,
    PopularityTitleBaseline,
    run_baseline_observation,
)
from storyflow.observation import (
    build_observation_input_records,
    read_jsonl,
    write_observation_inputs,
    write_jsonl,
)


def _workspace_tmp(name: str) -> Path:
    path = Path("outputs") / "test_tmp" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_processed_fixture(processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    with (processed_dir / "item_catalog.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "item_id",
                "title",
                "title_normalized",
                "popularity",
                "popularity_bucket",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "item_id": "item-head",
                    "title": "Head Serum",
                    "title_normalized": "head serum",
                    "popularity": 100,
                    "popularity_bucket": "head",
                },
                {
                    "item_id": "item-mid",
                    "title": "Mid Cleanser",
                    "title_normalized": "mid cleanser",
                    "popularity": 30,
                    "popularity_bucket": "mid",
                },
                {
                    "item_id": "item-tail",
                    "title": "Tail Balm",
                    "title_normalized": "tail balm",
                    "popularity": 3,
                    "popularity_bucket": "tail",
                },
            ],
        )
    write_jsonl(
        processed_dir / "observation_examples.jsonl",
        [
            {
                "example_id": "train-1",
                "user_id": "u-train-1",
                "history_item_ids": ["item-mid"],
                "history_item_titles": ["Mid Cleanser"],
                "history_timestamps": [1],
                "history_length": 1,
                "target_item_id": "item-tail",
                "target_title": "Tail Balm",
                "target_item_title": "Tail Balm",
                "target_timestamp": 2,
                "target_item_popularity": 3,
                "target_popularity_bucket": "tail",
                "split": "train",
            },
            {
                "example_id": "train-2",
                "user_id": "u-train-2",
                "history_item_ids": ["item-mid"],
                "history_item_titles": ["Mid Cleanser"],
                "history_timestamps": [3],
                "history_length": 1,
                "target_item_id": "item-tail",
                "target_title": "Tail Balm",
                "target_item_title": "Tail Balm",
                "target_timestamp": 4,
                "target_item_popularity": 3,
                "target_popularity_bucket": "tail",
                "split": "train",
            },
            {
                "example_id": "test-1",
                "user_id": "u-test-1",
                "history_item_ids": ["item-mid"],
                "history_item_titles": ["Mid Cleanser"],
                "history_timestamps": [5],
                "history_length": 1,
                "target_item_id": "item-tail",
                "target_title": "Tail Balm",
                "target_item_title": "Tail Balm",
                "target_timestamp": 6,
                "target_item_popularity": 3,
                "target_popularity_bucket": "tail",
                "split": "test",
            },
            {
                "example_id": "test-2",
                "user_id": "u-test-2",
                "history_item_ids": ["item-mid", "item-tail"],
                "history_item_titles": ["Mid Cleanser", "Tail Balm"],
                "history_timestamps": [7, 8],
                "history_length": 2,
                "target_item_id": "item-head",
                "target_title": "Head Serum",
                "target_item_title": "Head Serum",
                "target_timestamp": 9,
                "target_item_popularity": 100,
                "target_popularity_bucket": "head",
                "split": "test",
            },
        ],
    )


def _input_jsonl(workspace: Path, processed_dir: Path) -> Path:
    records = build_observation_input_records(
        dataset="synthetic_fixture",
        processed_suffix="tiny",
        split="test",
        processed_dir=processed_dir,
    )
    input_jsonl = workspace / "inputs.jsonl"
    write_observation_inputs(
        records,
        output_jsonl=input_jsonl,
        dataset="synthetic_fixture",
        processed_suffix="tiny",
        split="test",
        prompt_template="forced_json",
        stratify_by_popularity=False,
    )
    return input_jsonl


def test_popularity_baseline_predicts_most_popular_unseen_title() -> None:
    catalog = [
        {"item_id": "item-head", "title": "Head Serum", "popularity": 100},
        {"item_id": "item-mid", "title": "Mid Cleanser", "popularity": 30},
    ]
    baseline = PopularityTitleBaseline(catalog)

    output = baseline.predict({"history_item_ids": ["item-mid"]})

    assert output.generated_title == "Head Serum"
    assert output.selected_item_id == "item-head"
    assert output.score_source == "catalog_popularity"
    assert 0.0 < output.confidence <= 0.95


def test_cooccurrence_baseline_uses_train_split_counts() -> None:
    workspace = _workspace_tmp("cooccurrence_baseline")
    processed_dir = workspace / "processed"
    _write_processed_fixture(processed_dir)
    catalog_rows = list(csv.DictReader((processed_dir / "item_catalog.csv").open(encoding="utf-8")))
    baseline = CooccurrenceTitleBaseline(
        catalog_rows=catalog_rows,
        observation_examples_jsonl=processed_dir / "observation_examples.jsonl",
    )

    output = baseline.predict({"history_item_ids": ["item-mid"]})

    assert output.generated_title == "Tail Balm"
    assert output.selected_item_id == "item-tail"
    assert output.score_source == "train_split_cooccurrence"


def test_run_baseline_observation_writes_schema_and_resumes() -> None:
    workspace = _workspace_tmp("baseline_runner")
    processed_dir = workspace / "processed"
    _write_processed_fixture(processed_dir)
    input_jsonl = _input_jsonl(workspace, processed_dir)
    output_dir = workspace / "baseline_run"

    first = run_baseline_observation(
        input_jsonl=input_jsonl,
        output_dir=output_dir,
        baseline="cooccurrence",
        max_examples=1,
    )
    second = run_baseline_observation(
        input_jsonl=input_jsonl,
        output_dir=output_dir,
        baseline="cooccurrence",
        max_examples=2,
    )
    grounded = read_jsonl(output_dir / "grounded_predictions.jsonl")
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))

    assert first["newly_processed_count"] == 1
    assert second["newly_processed_count"] == 1
    assert len(grounded) == 2
    assert len({row["input_id"] for row in grounded}) == 2
    assert grounded[0]["provider"] == "baseline"
    assert grounded[0]["baseline"] == "cooccurrence"
    assert grounded[0]["api_called"] is False
    assert metrics["provider"] == "baseline"
    assert metrics["baseline"] == "cooccurrence"
    assert (output_dir / "raw_responses.jsonl").exists()
    assert (output_dir / "parsed_predictions.jsonl").exists()
    assert (output_dir / "report.md").exists()


def test_baseline_observation_cli_does_not_require_api_key(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    workspace = _workspace_tmp("baseline_cli")
    processed_dir = workspace / "processed"
    _write_processed_fixture(processed_dir)
    input_jsonl = _input_jsonl(workspace, processed_dir)
    output_dir = workspace / "cli_run"

    code = baseline_main(
        [
            "--input-jsonl",
            str(input_jsonl),
            "--baseline",
            "popularity",
            "--output-dir",
            str(output_dir),
            "--max-examples",
            "2",
            "--no-resume",
        ]
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert code == 0
    assert manifest["provider"] == "baseline"
    assert manifest["baseline"] == "popularity"
    assert manifest["api_called"] is False

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

from storyflow.generation import build_forced_json_prompt
from storyflow.observation import (
    build_observation_input_records,
    compute_observation_metrics,
    read_jsonl,
    run_mock_observation,
    write_observation_inputs,
    write_jsonl,
)
from storyflow.providers import MockProvider, parse_provider_response


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
                "genres",
                "popularity",
                "popularity_bucket",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "item_id": "item-head",
                    "title": "The Matrix",
                    "title_normalized": "matrix",
                    "genres": "Sci-Fi",
                    "popularity": 100,
                    "popularity_bucket": "head",
                },
                {
                    "item_id": "item-mid",
                    "title": "Arrival",
                    "title_normalized": "arrival",
                    "genres": "Sci-Fi",
                    "popularity": 30,
                    "popularity_bucket": "mid",
                },
                {
                    "item_id": "item-tail",
                    "title": "Primer",
                    "title_normalized": "primer",
                    "genres": "Sci-Fi",
                    "popularity": 3,
                    "popularity_bucket": "tail",
                },
            ]
        )
    write_jsonl(
        processed_dir / "observation_examples.jsonl",
        [
            {
                "example_id": "u1:2",
                "user_id": "u1",
                "history_item_ids": ["item-tail", "item-mid"],
                "history_item_titles": ["Primer", "Arrival"],
                "history_timestamps": [1, 2],
                "history_length": 2,
                "target_item_id": "item-head",
                "target_title": "The Matrix",
                "target_item_title": "The Matrix",
                "target_timestamp": 3,
                "target_item_popularity": 100,
                "target_popularity_bucket": "head",
                "split": "test",
            },
            {
                "example_id": "u2:2",
                "user_id": "u2",
                "history_item_ids": ["item-tail", "item-head"],
                "history_item_titles": ["Primer", "The Matrix"],
                "history_timestamps": [1, 2],
                "history_length": 2,
                "target_item_id": "item-mid",
                "target_title": "Arrival",
                "target_item_title": "Arrival",
                "target_timestamp": 4,
                "target_item_popularity": 30,
                "target_popularity_bucket": "mid",
                "split": "test",
            },
            {
                "example_id": "u3:2",
                "user_id": "u3",
                "history_item_ids": ["item-head", "item-mid"],
                "history_item_titles": ["The Matrix", "Arrival"],
                "history_timestamps": [1, 2],
                "history_length": 2,
                "target_item_id": "item-tail",
                "target_title": "Primer",
                "target_item_title": "Primer",
                "target_timestamp": 5,
                "target_item_popularity": 3,
                "target_popularity_bucket": "tail",
                "split": "test",
            },
        ],
    )


def test_forced_json_prompt_is_generative_title_task() -> None:
    prompt = build_forced_json_prompt(["Primer", "Arrival"])

    assert "title-level generative recommendation" in prompt
    assert "not ranking" in prompt
    assert "generated_title" in prompt
    assert "confidence" in prompt


def test_observation_input_records_have_prompt_schema() -> None:
    workspace = _workspace_tmp("observation_inputs")
    processed_dir = workspace / "processed"
    _write_processed_fixture(processed_dir)

    records = build_observation_input_records(
        dataset="synthetic_fixture",
        processed_suffix="tiny",
        split="test",
        processed_dir=processed_dir,
        max_examples=2,
        stratify_by_popularity=True,
    )

    assert len(records) == 2
    assert records[0]["history_item_titles"]
    assert records[0]["target_title"]
    assert records[0]["prompt_hash"]
    assert records[0]["target_popularity_bucket"] in {"head", "mid", "tail"}

    output_jsonl = workspace / "inputs.jsonl"
    manifest = write_observation_inputs(
        records,
        output_jsonl=output_jsonl,
        dataset="synthetic_fixture",
        processed_suffix="tiny",
        split="test",
        prompt_template="forced_json",
        stratify_by_popularity=True,
    )
    assert output_jsonl.exists()
    assert manifest["input_count"] == 2


def test_mock_provider_is_deterministic_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    catalog = [
        {"item_id": "item-head", "title": "The Matrix", "popularity": 100},
        {"item_id": "item-tail", "title": "Primer", "popularity": 3},
    ]
    input_record = {
        "input_id": "input-1",
        "prompt_hash": "hash-1",
        "target_title": "Primer",
    }
    provider = MockProvider(catalog, mode="oracle-ish", seed=7)

    first = provider.generate(input_record)
    second = provider.generate(input_record)

    assert first.raw_text == second.raw_text
    assert first.provider == "mock"


def test_provider_response_parser_accepts_percentage_confidence() -> None:
    parsed = parse_provider_response(
        json.dumps(
            {
                "generated_title": "Primer",
                "is_likely_correct": "yes",
                "confidence": "72%",
            }
        )
    )

    assert parsed.generated_title == "Primer"
    assert parsed.confidence == 0.72
    assert parsed.is_likely_correct == "yes"


def test_mock_observation_runner_grounding_metrics_and_resume() -> None:
    workspace = _workspace_tmp("mock_runner")
    processed_dir = workspace / "processed"
    _write_processed_fixture(processed_dir)
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
    output_dir = workspace / "mock_run"

    first_manifest = run_mock_observation(
        input_jsonl=input_jsonl,
        output_dir=output_dir,
        provider_mode="oracle-ish",
        max_examples=2,
    )
    second_manifest = run_mock_observation(
        input_jsonl=input_jsonl,
        output_dir=output_dir,
        provider_mode="oracle-ish",
        max_examples=3,
    )

    grounded = read_jsonl(output_dir / "grounded_predictions.jsonl")
    assert first_manifest["newly_processed_count"] == 2
    assert second_manifest["newly_processed_count"] == 1
    assert len(grounded) == 3
    assert len({row["input_id"] for row in grounded}) == 3
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["provider"] == "mock"
    assert "bucket_metrics" in metrics
    assert (output_dir / "report.md").exists()


def test_observation_metrics_report_counts_cases() -> None:
    metrics = compute_observation_metrics(
        [
            {
                "confidence": 0.9,
                "correctness": 0,
                "target_popularity_bucket": "head",
                "grounded_item_id": "item-head",
            },
            {
                "confidence": 0.4,
                "correctness": 1,
                "target_popularity_bucket": "tail",
                "grounded_item_id": "item-tail",
            },
        ]
    )

    assert metrics["wrong_high_confidence_count"] == 1
    assert metrics["correct_low_confidence_count"] == 1
    assert metrics["bucket_metrics"]["head"]["count"] == 1

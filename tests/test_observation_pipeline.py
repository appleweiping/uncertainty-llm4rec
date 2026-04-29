from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

from scripts.build_observation_gate_inputs import main as gate_inputs_main
from storyflow.generation import (
    build_catalog_constrained_json_prompt,
    build_forced_json_prompt,
    build_retrieval_context_json_prompt,
)
from storyflow.observation import (
    build_observation_input_records,
    compute_observation_metrics,
    default_observation_input_path,
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
            {
                "example_id": "u4:2",
                "user_id": "u4",
                "history_item_ids": ["item-head", "item-mid"],
                "history_item_titles": ["The Matrix", "Arrival"],
                "history_timestamps": [1, 2],
                "history_length": 2,
                "target_item_id": "item-head",
                "target_title": "The Matrix",
                "target_item_title": "The Matrix",
                "target_timestamp": 6,
                "target_item_popularity": 100,
                "target_popularity_bucket": "head",
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


def test_catalog_constrained_prompt_is_diagnostic_grounding_gate() -> None:
    prompt = build_catalog_constrained_json_prompt(
        ["Primer", "Arrival"],
        ["The Matrix", "Toy Story"],
    )

    assert "catalog-grounding diagnostic" in prompt
    assert "Catalog candidate titles" in prompt
    assert "The Matrix" in prompt
    assert "NO_GROUNDABLE_TITLE" in prompt
    assert "confidence" in prompt


def test_retrieval_context_prompt_is_grounding_context_gate() -> None:
    prompt = build_retrieval_context_json_prompt(
        ["Primer", "Arrival"],
        ["The Matrix", "Solaris"],
    )

    assert "retrieved without using the held-out target item" in prompt
    assert "grounding context" in prompt
    assert "NO_GROUNDABLE_TITLE" in prompt
    assert "generated_title" in prompt


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
    assert manifest["repeat_counts"]["target_in_history_count"] == 0


def test_observation_input_records_support_repeat_target_policies() -> None:
    workspace = _workspace_tmp("repeat_target_inputs")
    processed_dir = workspace / "processed"
    _write_processed_fixture(processed_dir)

    all_records = build_observation_input_records(
        dataset="synthetic_fixture",
        processed_suffix="tiny",
        split="test",
        processed_dir=processed_dir,
    )
    no_repeat_records = build_observation_input_records(
        dataset="synthetic_fixture",
        processed_suffix="tiny",
        split="test",
        processed_dir=processed_dir,
        repeat_target_policy="exclude",
    )
    repeat_only_records = build_observation_input_records(
        dataset="synthetic_fixture",
        processed_suffix="tiny",
        split="test",
        processed_dir=processed_dir,
        repeat_target_policy="only",
    )

    assert len(all_records) == 4
    assert len(no_repeat_records) == 3
    assert len(repeat_only_records) == 1
    assert repeat_only_records[0]["target_in_history"] is True
    assert repeat_only_records[0]["target_history_occurrence_count"] == 1
    assert repeat_only_records[0]["repeat_target_policy"] == "only"
    assert "no_repeat" in str(
        default_observation_input_path(
            dataset="synthetic_fixture",
            processed_suffix="tiny",
            split="test",
            prompt_template="forced_json",
            repeat_target_policy="exclude",
        )
    )


def test_catalog_constrained_observation_inputs_exclude_target_by_default() -> None:
    workspace = _workspace_tmp("constrained_observation_inputs")
    processed_dir = workspace / "processed"
    _write_processed_fixture(processed_dir)

    records = build_observation_input_records(
        dataset="synthetic_fixture",
        processed_suffix="tiny",
        split="test",
        processed_dir=processed_dir,
        max_examples=1,
        prompt_template="catalog_constrained_json",
        candidate_count=2,
    )

    assert len(records) == 1
    record = records[0]
    assert record["prompt_template"] == "catalog_constrained_json"
    assert record["catalog_candidate_titles"]
    assert record["target_item_id"] not in record["catalog_candidate_item_ids"]
    assert record["candidate_policy"]["target_in_candidates"] is False
    assert record["candidate_policy"]["is_diagnostic_grounding_gate"] is True
    assert "Catalog candidate titles" in record["prompt"]


def test_retrieval_context_observation_inputs_use_history_overlap_policy() -> None:
    workspace = _workspace_tmp("retrieval_context_observation_inputs")
    processed_dir = workspace / "processed"
    _write_processed_fixture(processed_dir)

    records = build_observation_input_records(
        dataset="synthetic_fixture",
        processed_suffix="tiny",
        split="test",
        processed_dir=processed_dir,
        max_examples=1,
        prompt_template="retrieval_context_json",
        candidate_count=2,
    )

    assert len(records) == 1
    record = records[0]
    assert record["prompt_template"] == "retrieval_context_json"
    assert record["target_item_id"] not in record["catalog_candidate_item_ids"]
    assert record["candidate_policy"]["candidate_policy"] == "history_token_overlap"
    assert record["candidate_policy"]["is_retrieval_context_gate"] is True
    assert len(record["catalog_candidate_scores"]) == len(record["catalog_candidate_titles"])
    assert "grounding context" in record["prompt"]


def test_observation_gate_script_writes_three_input_variants() -> None:
    workspace = _workspace_tmp("observation_gate_script")
    processed_dir = workspace / "processed"
    output_manifest = workspace / "gate_manifest.json"
    _write_processed_fixture(processed_dir)

    code = gate_inputs_main(
        [
            "--dataset",
            "synthetic_fixture",
            "--processed-suffix",
            "tiny",
            "--processed-dir",
            str(processed_dir),
            "--split",
            "test",
            "--max-examples",
            "2",
            "--candidate-count",
            "2",
            "--stratify-by-popularity",
            "--output-manifest",
            str(output_manifest),
        ]
    )

    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
    variants = {variant["prompt_template"]: variant for variant in manifest["variants"]}
    assert code == 0
    assert manifest["api_called"] is False
    assert set(variants) == {
        "forced_json",
        "catalog_constrained_json",
        "retrieval_context_json",
    }
    assert variants["forced_json"]["candidate_summary"]["candidate_record_count"] == 0
    assert variants["catalog_constrained_json"]["candidate_summary"]["target_leak_count"] == 0
    assert variants["retrieval_context_json"]["candidate_policy"] == "history_token_overlap"
    assert Path(variants["forced_json"]["input_jsonl"]).name == "test_gate2_forced_json.jsonl"
    assert (
        Path(variants["catalog_constrained_json"]["input_jsonl"]).name
        == "test_gate2_catalog_constrained_json_c2.jsonl"
    )
    assert (
        Path(variants["retrieval_context_json"]["input_jsonl"]).name
        == "test_gate2_retrieval_context_json_c2.jsonl"
    )


def test_observation_gate_script_supports_repeat_free_variants() -> None:
    workspace = _workspace_tmp("observation_gate_no_repeat_script")
    processed_dir = workspace / "processed"
    output_manifest = workspace / "gate_manifest.json"
    _write_processed_fixture(processed_dir)

    code = gate_inputs_main(
        [
            "--dataset",
            "synthetic_fixture",
            "--processed-suffix",
            "tiny",
            "--processed-dir",
            str(processed_dir),
            "--split",
            "test",
            "--max-examples",
            "3",
            "--candidate-count",
            "2",
            "--stratify-by-popularity",
            "--repeat-target-policy",
            "exclude",
            "--output-manifest",
            str(output_manifest),
        ]
    )

    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
    variants = {variant["prompt_template"]: variant for variant in manifest["variants"]}
    assert code == 0
    assert manifest["repeat_target_policy"] == "exclude"
    assert {variant["input_count"] for variant in variants.values()} == {3}
    assert {
        variant["repeat_counts"]["target_in_history_count"]
        for variant in variants.values()
    } == {0}
    assert variants["catalog_constrained_json"]["candidate_summary"]["target_leak_count"] == 0
    assert variants["retrieval_context_json"]["candidate_summary"]["target_leak_count"] == 0
    assert Path(variants["forced_json"]["input_jsonl"]).name == (
        "test_gate3_no_repeat_forced_json.jsonl"
    )
    assert (
        Path(variants["retrieval_context_json"]["input_jsonl"]).name
        == "test_gate3_no_repeat_retrieval_context_json_c2.jsonl"
    )


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

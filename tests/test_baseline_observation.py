from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

import pytest

from scripts.analyze_observation import main as analyze_main
from scripts.run_baseline_observation import main as baseline_main
from scripts.validate_baseline_artifact import main as validate_baseline_main
from storyflow.baselines import (
    CooccurrenceTitleBaseline,
    PopularityTitleBaseline,
    RankingJsonlTitleBaseline,
    parse_ranking_candidates,
    run_baseline_observation,
    validate_baseline_artifact,
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


def test_ranking_candidate_parser_accepts_common_shapes() -> None:
    list_shape = parse_ranking_candidates(
        {"input_id": "x", "ranked_item_ids": ["a", "b"], "scores": [3.0, 1.0]}
    )
    dict_shape = parse_ranking_candidates(
        {"input_id": "x", "ranked_items": [{"item_id": "c", "score": "2.5"}]}
    )

    assert [candidate.item_id for candidate in list_shape] == ["a", "b"]
    assert list_shape[0].score == 3.0
    assert dict_shape[0].item_id == "c"
    assert dict_shape[0].score == 2.5


def test_ranking_jsonl_baseline_filters_history_and_uses_catalog_title() -> None:
    workspace = _workspace_tmp("ranking_jsonl_baseline")
    processed_dir = workspace / "processed"
    _write_processed_fixture(processed_dir)
    input_jsonl = _input_jsonl(workspace, processed_dir)
    inputs = read_jsonl(input_jsonl)
    ranking_jsonl = workspace / "ranking.jsonl"
    write_jsonl(
        ranking_jsonl,
        [
            {
                "input_id": inputs[0]["input_id"],
                "ranked_item_ids": ["item-mid", "item-tail", "item-head"],
                "scores": [10.0, 4.0, 1.0],
                "run_id": "sasrec-fixture",
            }
        ],
    )
    catalog_rows = list(csv.DictReader((processed_dir / "item_catalog.csv").open(encoding="utf-8")))
    baseline = RankingJsonlTitleBaseline(
        catalog_rows=catalog_rows,
        ranking_jsonl=ranking_jsonl,
        fallback_to_popularity=False,
    )

    output = baseline.predict(inputs[0])

    assert output.generated_title == "Tail Balm"
    assert output.selected_item_id == "item-tail"
    assert output.selected_rank == 2
    assert output.candidate_count == 3
    assert output.source_record_id == "sasrec-fixture"
    assert output.score_source == "ranking_jsonl_softmax_score"


def test_ranking_jsonl_baseline_strict_mode_rejects_missing_prediction() -> None:
    workspace = _workspace_tmp("ranking_jsonl_missing")
    processed_dir = workspace / "processed"
    _write_processed_fixture(processed_dir)
    input_jsonl = _input_jsonl(workspace, processed_dir)
    inputs = read_jsonl(input_jsonl)
    ranking_jsonl = workspace / "ranking.jsonl"
    write_jsonl(
        ranking_jsonl,
        [{"input_id": "different-input", "ranked_item_ids": ["item-head"]}],
    )
    catalog_rows = list(csv.DictReader((processed_dir / "item_catalog.csv").open(encoding="utf-8")))
    baseline = RankingJsonlTitleBaseline(
        catalog_rows=catalog_rows,
        ranking_jsonl=ranking_jsonl,
        fallback_to_popularity=False,
    )

    with pytest.raises(ValueError, match="missing_input"):
        baseline.predict(inputs[0])


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


def test_baseline_observation_analysis_marks_proxy_confidence() -> None:
    workspace = _workspace_tmp("baseline_analysis")
    processed_dir = workspace / "processed"
    _write_processed_fixture(processed_dir)
    input_jsonl = _input_jsonl(workspace, processed_dir)
    output_dir = workspace / "baseline_run"
    analysis_dir = workspace / "analysis"
    registry_jsonl = workspace / "registry.jsonl"
    run_baseline_observation(
        input_jsonl=input_jsonl,
        output_dir=output_dir,
        baseline="cooccurrence",
        max_examples=2,
    )

    code = analyze_main(
        [
            "--run-dir",
            str(output_dir),
            "--output-dir",
            str(analysis_dir),
            "--registry-jsonl",
            str(registry_jsonl),
            "--source-label",
            "baseline-cooccurrence-fixture",
        ]
    )
    summary = json.loads((analysis_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((analysis_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    registry_rows = read_jsonl(registry_jsonl)

    assert code == 0
    assert summary["provider"] == "baseline"
    assert summary["baseline"] == "cooccurrence"
    assert summary["source_profile"]["source_kind"] == "baseline_observation"
    assert summary["claim_guardrails"]["baseline_confidence_is_proxy"] is True
    assert summary["claim_guardrails"]["confidence_is_calibrated"] is False
    assert manifest["source_kind"] == "baseline_observation"
    assert registry_rows[0]["source_kind"] == "baseline_observation"
    assert registry_rows[0]["confidence_semantics"] == "non_calibrated_baseline_proxy"


def test_run_ranking_jsonl_observation_writes_adapter_metadata() -> None:
    workspace = _workspace_tmp("ranking_jsonl_runner")
    processed_dir = workspace / "processed"
    _write_processed_fixture(processed_dir)
    input_jsonl = _input_jsonl(workspace, processed_dir)
    inputs = read_jsonl(input_jsonl)
    ranking_jsonl = workspace / "ranking.jsonl"
    write_jsonl(
        ranking_jsonl,
        [
            {
                "input_id": inputs[0]["input_id"],
                "ranked_items": [
                    {"item_id": "item-mid", "score": 7.0},
                    {"item_id": "item-tail", "score": 3.0},
                ],
            },
            {
                "input_id": inputs[1]["input_id"],
                "ranked_items": [{"item_id": "item-head", "score": 5.0}],
            },
        ],
    )
    output_dir = workspace / "ranking_run"

    manifest = run_baseline_observation(
        input_jsonl=input_jsonl,
        output_dir=output_dir,
        baseline="ranking_jsonl",
        ranking_jsonl=ranking_jsonl,
        max_examples=2,
    )
    grounded = read_jsonl(output_dir / "grounded_predictions.jsonl")

    assert manifest["baseline"] == "ranking_jsonl"
    assert manifest["ranking_jsonl"] == str(ranking_jsonl)
    assert grounded[0]["generated_title"] == "Tail Balm"
    assert grounded[0]["baseline_selected_rank"] == 2
    assert grounded[0]["baseline_candidate_count"] == 2
    assert grounded[0]["api_called"] is False


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


def test_baseline_observation_cli_runs_ranking_jsonl_adapter(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    workspace = _workspace_tmp("baseline_cli_ranking")
    processed_dir = workspace / "processed"
    _write_processed_fixture(processed_dir)
    input_jsonl = _input_jsonl(workspace, processed_dir)
    inputs = read_jsonl(input_jsonl)
    ranking_jsonl = workspace / "ranking.jsonl"
    write_jsonl(
        ranking_jsonl,
        [
            {"input_id": inputs[0]["input_id"], "ranked_item_ids": ["item-tail"]},
            {"input_id": inputs[1]["input_id"], "ranked_item_ids": ["item-head"]},
        ],
    )
    output_dir = workspace / "cli_ranking_run"

    code = baseline_main(
        [
            "--input-jsonl",
            str(input_jsonl),
            "--baseline",
            "ranking_jsonl",
            "--ranking-jsonl",
            str(ranking_jsonl),
            "--output-dir",
            str(output_dir),
            "--max-examples",
            "2",
            "--strict-ranking",
            "--no-resume",
        ]
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert code == 0
    assert manifest["baseline"] == "ranking_jsonl"
    assert manifest["strict_ranking"] is True
    assert manifest["api_called"] is False


def test_validate_baseline_artifact_writes_pass_manifest() -> None:
    workspace = _workspace_tmp("baseline_artifact_valid")
    processed_dir = workspace / "processed"
    _write_processed_fixture(processed_dir)
    input_jsonl = _input_jsonl(workspace, processed_dir)
    inputs = read_jsonl(input_jsonl)
    ranking_jsonl = workspace / "ranking.jsonl"
    output_manifest = workspace / "validation_manifest.json"
    write_jsonl(
        ranking_jsonl,
        [
            {
                "input_id": inputs[0]["input_id"],
                "ranked_item_ids": ["item-tail", "item-head"],
                "scores": [2.0, 1.0],
                "run_id": "sasrec-fixture",
            },
            {
                "input_id": inputs[1]["input_id"],
                "ranked_items": [{"item_id": "item-head", "score": 4.0}],
                "run_id": "sasrec-fixture",
            },
        ],
    )

    manifest = validate_baseline_artifact(
        ranking_jsonl=ranking_jsonl,
        input_jsonl=input_jsonl,
        baseline_family="sasrec",
        model_family="SASRec",
        run_label="sasrec-fixture",
        dataset="synthetic_fixture",
        processed_suffix="tiny",
        split="test",
        trained_splits=["train"],
        seed=7,
        output_manifest_json=output_manifest,
        strict=True,
    )

    assert manifest["validation_status"] == "passed"
    assert manifest["coverage"]["selected_input_count"] == 2
    assert manifest["coverage"]["missing_input_count"] == 0
    assert manifest["quality"]["target_item_in_ranking_count"] == 2
    assert manifest["validator_api_called"] is False
    assert manifest["validator_model_training"] is False
    assert manifest["validator_server_executed"] is False
    assert manifest["is_experiment_result"] is False
    assert output_manifest.exists()


def test_validate_baseline_artifact_flags_schema_and_coverage_failures() -> None:
    workspace = _workspace_tmp("baseline_artifact_invalid")
    processed_dir = workspace / "processed"
    _write_processed_fixture(processed_dir)
    input_jsonl = _input_jsonl(workspace, processed_dir)
    inputs = read_jsonl(input_jsonl)
    ranking_jsonl = workspace / "ranking.jsonl"
    write_jsonl(
        ranking_jsonl,
        [
            {
                "input_id": inputs[0]["input_id"],
                "ranked_item_ids": ["item-tail", "item-tail", "unknown-item"],
                "scores": [1.0],
            }
        ],
    )

    manifest = validate_baseline_artifact(
        ranking_jsonl=ranking_jsonl,
        input_jsonl=input_jsonl,
        baseline_family="sasrec",
        strict=True,
    )
    error_codes = manifest["problem_summary"]["errors"]

    assert manifest["validation_status"] == "failed"
    assert error_codes["missing_input_ranking"] == 1
    assert error_codes["score_length_mismatch"] == 1
    assert error_codes["duplicate_candidate_item_id"] == 1
    assert error_codes["unknown_catalog_item_id"] == 1


def test_validate_baseline_artifact_cli_does_not_require_api_key(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    workspace = _workspace_tmp("baseline_artifact_cli")
    processed_dir = workspace / "processed"
    _write_processed_fixture(processed_dir)
    input_jsonl = _input_jsonl(workspace, processed_dir)
    inputs = read_jsonl(input_jsonl)
    ranking_jsonl = workspace / "ranking.jsonl"
    output_manifest = workspace / "cli_validation_manifest.json"
    write_jsonl(
        ranking_jsonl,
        [
            {"input_id": inputs[0]["input_id"], "ranked_item_ids": ["item-tail"]},
            {"input_id": inputs[1]["input_id"], "ranked_item_ids": ["item-head"]},
        ],
    )

    code = validate_baseline_main(
        [
            "--ranking-jsonl",
            str(ranking_jsonl),
            "--input-jsonl",
            str(input_jsonl),
            "--baseline-family",
            "sasrec",
            "--model-family",
            "SASRec",
            "--run-label",
            "sasrec-fixture",
            "--dataset",
            "synthetic_fixture",
            "--processed-suffix",
            "tiny",
            "--split",
            "test",
            "--trained-splits",
            "train",
            "--output-manifest-json",
            str(output_manifest),
            "--strict",
        ]
    )
    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))

    assert code == 0
    assert manifest["validation_status"] == "passed"
    assert manifest["baseline_family"] == "sasrec"
    assert manifest["validator_api_called"] is False

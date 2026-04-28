from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

from scripts.analyze_observation import main as analyze_main
from scripts.review_observation_cases import main as review_main
from storyflow.analysis import (
    analyze_observation_run,
    append_registry_record,
    popularity_confidence_slope,
    reliability_bins,
    review_observation_cases,
    summarize_observation_records,
)
from storyflow.observation import read_jsonl, write_jsonl


def _workspace(name: str) -> Path:
    path = Path("outputs") / "test_tmp" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _grounded_rows() -> list[dict[str, object]]:
    return [
        {
            "input_id": "a",
            "example_id": "u1:1",
            "user_id": "u1",
            "provider": "mock",
            "model": "fixture",
            "dry_run": True,
            "generated_title": "Head Hit",
            "target_title": "Head Hit",
            "confidence": 0.9,
            "correctness": 1,
            "target_popularity": 100,
            "target_popularity_bucket": "head",
            "grounded_item_id": "i1",
            "grounding_status": "exact",
            "grounding_score": 1.0,
            "parse_strategy": "strict_json",
        },
        {
            "input_id": "b",
            "example_id": "u2:1",
            "user_id": "u2",
            "provider": "mock",
            "model": "fixture",
            "dry_run": True,
            "generated_title": "Popular Wrong",
            "target_title": "Tail Truth",
            "confidence": 0.84,
            "correctness": 0,
            "target_popularity": 80,
            "target_popularity_bucket": "head",
            "grounded_item_id": "i2",
            "grounding_status": "normalized_exact",
            "grounding_score": 0.98,
            "parse_strategy": "fenced_json",
        },
        {
            "input_id": "c",
            "example_id": "u3:1",
            "user_id": "u3",
            "provider": "mock",
            "model": "fixture",
            "dry_run": True,
            "generated_title": "Tail Hit",
            "target_title": "Tail Hit",
            "confidence": 0.3,
            "correctness": 1,
            "target_popularity": 2,
            "target_popularity_bucket": "tail",
            "grounded_item_id": "i3",
            "grounding_status": "exact",
            "grounding_score": 1.0,
            "parse_strategy": "regex",
        },
        {
            "input_id": "d",
            "example_id": "u4:1",
            "user_id": "u4",
            "provider": "mock",
            "model": "fixture",
            "dry_run": True,
            "generated_title": "No Catalog",
            "target_title": "Mid Truth",
            "confidence": 0.2,
            "correctness": 0,
            "target_popularity": 10,
            "target_popularity_bucket": "mid",
            "grounded_item_id": None,
            "grounding_status": "out_of_catalog",
            "grounding_score": 0.0,
            "parse_strategy": "strict_json",
        },
    ]


def test_reliability_and_summary_slices() -> None:
    rows = _grounded_rows()
    failed = [
        {
            "input_id": "e",
            "failure_stage": "parse",
            "error": "missing generated title",
        }
    ]

    bins = reliability_bins(rows, n_bins=5)
    summary = summarize_observation_records(rows, failed_rows=failed, max_cases=3)

    assert len(bins) == 5
    assert summary["count"] == 4
    assert summary["parse_failure_count"] == 1
    assert summary["grounding_summary"]["failure_count"] == 1
    assert summary["quadrant_counts"]["wrong_high_confidence"] == 1
    assert summary["quadrant_counts"]["correct_low_confidence"] == 1
    assert summary["bucket_summary"]["head"]["count"] == 2
    assert summary["risk_cases"]["wrong_high_confidence"][0]["input_id"] == "b"
    assert summary["risk_cases"]["correct_low_confidence"][0]["input_id"] == "c"


def test_popularity_confidence_slope_has_controlled_field() -> None:
    slope = popularity_confidence_slope(_grounded_rows())

    assert slope["n"] == 4
    assert "univariate" in slope
    assert "correctness_residualized" in slope
    assert slope["univariate"]["slope"] is not None


def test_analyze_observation_run_writes_outputs_and_registry() -> None:
    workspace = _workspace("analysis")
    run_dir = workspace / "run"
    run_dir.mkdir()
    grounded = run_dir / "grounded_predictions.jsonl"
    failed = run_dir / "failed_cases.jsonl"
    manifest = run_dir / "manifest.json"
    write_jsonl(grounded, _grounded_rows())
    write_jsonl(
        failed,
        [{"input_id": "e", "failure_stage": "parse", "error": "bad json"}],
    )
    manifest.write_text(
        json.dumps(
            {
                "provider": "mock",
                "model": "fixture",
                "dry_run": True,
                "api_called": False,
                "output_dir": str(run_dir),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    analysis_manifest = analyze_observation_run(
        grounded_jsonl=grounded,
        failed_jsonl=failed,
        manifest_json=manifest,
        output_dir=workspace / "analysis",
    )
    registry_record = append_registry_record(
        registry_jsonl=workspace / "registry.jsonl",
        analysis_manifest=analysis_manifest,
        source_label="unit-test",
    )

    summary = json.loads(Path(analysis_manifest["summary"]).read_text(encoding="utf-8"))
    risk_rows = read_jsonl(analysis_manifest["risk_cases"])
    registry_rows = read_jsonl(workspace / "registry.jsonl")
    assert summary["provider"] == "mock"
    assert summary["api_called"] is False
    assert any(row["slice"] == "wrong_high_confidence" for row in risk_rows)
    assert registry_rows[0]["run_id"] == registry_record["run_id"]


def test_analyze_observation_cli_run_dir() -> None:
    workspace = _workspace("analysis_cli")
    run_dir = workspace / "run"
    output_dir = workspace / "out"
    run_dir.mkdir()
    write_jsonl(run_dir / "grounded_predictions.jsonl", _grounded_rows())
    (run_dir / "manifest.json").write_text(
        json.dumps({"provider": "mock", "dry_run": True, "api_called": False}),
        encoding="utf-8",
    )

    code = analyze_main(
        [
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
            "--registry-jsonl",
            str(workspace / "registry.jsonl"),
            "--source-label",
            "cli-test",
        ]
    )

    assert code == 0
    assert (output_dir / "analysis_summary.json").exists()
    assert read_jsonl(workspace / "registry.jsonl")[0]["source_label"] == "cli-test"


def test_case_review_taxonomy_joins_history_and_catalog() -> None:
    workspace = _workspace("case_review")
    run_dir = workspace / "run"
    run_dir.mkdir()
    catalog = workspace / "item_catalog.csv"
    with catalog.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["item_id", "title", "title_normalized", "popularity", "popularity_bucket"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "item_id": "i2",
                "title": "Popular Wrong",
                "title_normalized": "popular wrong",
                "popularity": 200,
                "popularity_bucket": "head",
            }
        )
        writer.writerow(
            {
                "item_id": "i3",
                "title": "Tail Hit",
                "title_normalized": "tail hit",
                "popularity": 2,
                "popularity_bucket": "tail",
            }
        )
    input_jsonl = workspace / "inputs.jsonl"
    write_jsonl(
        input_jsonl,
        [
            {
                "input_id": "b",
                "user_id": "u2",
                "split": "test",
                "history_item_titles": ["A", "B", "C"],
                "history_length": 3,
                "target_item_id": "i3",
                "target_title": "Tail Truth",
                "target_popularity": 2,
                "target_popularity_bucket": "tail",
                "source": {"catalog_csv": str(catalog)},
            }
        ],
    )
    write_jsonl(run_dir / "grounded_predictions.jsonl", [_grounded_rows()[1]])
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "dry_run": False,
                "api_called": True,
                "input_jsonl": str(input_jsonl),
            }
        ),
        encoding="utf-8",
    )

    manifest = review_observation_cases(
        grounded_jsonl=run_dir / "grounded_predictions.jsonl",
        manifest_json=run_dir / "manifest.json",
        output_dir=workspace / "case_review",
    )

    summary = json.loads(Path(manifest["summary"]).read_text(encoding="utf-8"))
    cases = read_jsonl(manifest["cases"])
    assert summary["provider"] == "deepseek"
    assert summary["taxonomy_counts"]["wrong_high_confidence"] == 1
    assert "self_verified_wrong" not in summary["tag_counts"]
    assert "generated_more_popular_than_target" in cases[0]["taxonomy_tags"]
    assert cases[0]["history_titles_tail"] == ["A", "B", "C"]


def test_case_review_cli_run_dir() -> None:
    workspace = _workspace("case_review_cli")
    run_dir = workspace / "run"
    output_dir = workspace / "out"
    run_dir.mkdir()
    write_jsonl(run_dir / "grounded_predictions.jsonl", _grounded_rows())
    (run_dir / "manifest.json").write_text(
        json.dumps({"provider": "mock", "dry_run": True, "api_called": False}),
        encoding="utf-8",
    )

    code = review_main(
        [
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert code == 0
    assert (output_dir / "case_review_summary.json").exists()
    assert (output_dir / "case_review_cases.jsonl").exists()

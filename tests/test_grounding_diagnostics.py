from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

from scripts.analyze_grounding_diagnostics import main as diagnostics_main
from storyflow.analysis import (
    analyze_grounding_diagnostics,
    catalog_grounding_summary,
    duplicate_title_groups,
    grounding_margin_summary,
)
from storyflow.analysis.grounding_diagnostics import read_jsonl


def _workspace(name: str) -> Path:
    path = Path("outputs") / "test_tmp" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_catalog(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
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
                    "item_id": "a",
                    "title": "Tea Tree Shampoo",
                    "title_normalized": "tea tree shampoo",
                    "popularity": "20",
                    "popularity_bucket": "head",
                },
                {
                    "item_id": "b",
                    "title": "Tea-Tree Shampoo!",
                    "title_normalized": "tea tree shampoo",
                    "popularity": "5",
                    "popularity_bucket": "tail",
                },
                {
                    "item_id": "c",
                    "title": "Rose Lotion",
                    "title_normalized": "rose lotion",
                    "popularity": "10",
                    "popularity_bucket": "mid",
                },
            ]
        )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _grounded_rows() -> list[dict[str, object]]:
    return [
        {
            "input_id": "one",
            "example_id": "u1:1",
            "user_id": "u1",
            "generated_title": "Tea Tree Shampoo",
            "target_title": "Tea Tree Shampoo",
            "confidence": 0.91,
            "correctness": 1,
            "grounded_item_id": "a",
            "grounding_status": "exact",
            "grounding_score": 1.0,
            "grounding_second_score": 0.99,
            "grounding_ambiguity": 0.99,
            "target_popularity_bucket": "head",
            "grounding_candidates": [
                {"item_id": "a", "title": "Tea Tree Shampoo", "score": 1.0, "rank": 1},
                {"item_id": "b", "title": "Tea-Tree Shampoo!", "score": 0.99, "rank": 2},
            ],
        },
        {
            "input_id": "two",
            "example_id": "u2:1",
            "user_id": "u2",
            "generated_title": "Rose Lotion",
            "target_title": "Rose Lotion",
            "confidence": 0.6,
            "correctness": 1,
            "grounded_item_id": "c",
            "grounding_status": "exact",
            "grounding_score": 1.0,
            "grounding_second_score": 0.5,
            "grounding_ambiguity": 0.5,
            "target_popularity_bucket": "mid",
            "grounding_candidates": [
                {"item_id": "c", "title": "Rose Lotion", "score": 1.0, "rank": 1},
                {"item_id": "b", "title": "Tea-Tree Shampoo!", "score": 0.5, "rank": 2},
            ],
        },
    ]


def test_catalog_duplicate_title_groups() -> None:
    catalog = [
        {
            "item_id": "a",
            "title": "Tea Tree Shampoo",
            "title_normalized": "tea tree shampoo",
            "popularity": 20,
            "popularity_bucket": "head",
        },
        {
            "item_id": "b",
            "title": "Tea-Tree Shampoo!",
            "title_normalized": "tea tree shampoo",
            "popularity": 5,
            "popularity_bucket": "tail",
        },
        {
            "item_id": "c",
            "title": "Rose Lotion",
            "title_normalized": "rose lotion",
            "popularity": 10,
            "popularity_bucket": "mid",
        },
    ]

    summary = catalog_grounding_summary(catalog)
    duplicates = duplicate_title_groups(catalog)

    assert summary["duplicate_normalized_group_count"] == 1
    assert summary["duplicate_normalized_item_count"] == 2
    assert duplicates[0]["normalized_title"] == "tea tree shampoo"
    assert duplicates[0]["bucket_counts"] == {"head": 1, "tail": 1}


def test_grounding_margin_summary_extracts_low_margin_cases() -> None:
    summary = grounding_margin_summary(_grounded_rows(), margin_threshold=0.03)

    assert summary["grounded_row_count"] == 2
    assert summary["rows_with_top_two_scores"] == 2
    assert summary["low_margin_count"] == 1
    assert summary["low_margin_cases"][0]["input_id"] == "one"
    assert summary["low_margin_cases"][0]["grounding_margin"] < 0.03


def test_analyze_grounding_diagnostics_writes_artifacts() -> None:
    workspace = _workspace("grounding_diagnostics")
    catalog = workspace / "item_catalog.csv"
    grounded = workspace / "grounded_predictions.jsonl"
    manifest = workspace / "manifest.json"
    _write_catalog(catalog)
    _write_jsonl(grounded, _grounded_rows())
    manifest.write_text(
        json.dumps(
            {
                "provider": "mock",
                "model": "fixture",
                "dry_run": True,
                "api_called": False,
            }
        ),
        encoding="utf-8",
    )

    result = analyze_grounding_diagnostics(
        catalog_csv=catalog,
        grounded_jsonl=grounded,
        manifest_json=manifest,
        output_dir=workspace / "out",
        dataset="fixture",
        processed_suffix="tiny",
    )

    summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
    duplicates = read_jsonl(result["duplicate_title_groups"])
    low_margin = read_jsonl(result["low_margin_cases"])
    assert summary["catalog"]["duplicate_normalized_group_count"] == 1
    assert summary["observation_margins"]["low_margin_count"] == 1
    assert duplicates[0]["normalized_title"] == "tea tree shampoo"
    assert low_margin[0]["input_id"] == "one"


def test_grounding_diagnostics_cli_dataset_processed_suffix() -> None:
    workspace = _workspace("grounding_diagnostics_cli")
    processed = workspace / "data" / "processed" / "fixture" / "tiny"
    catalog = processed / "item_catalog.csv"
    output_dir = workspace / "out"
    _write_catalog(catalog)

    code = diagnostics_main(
        [
            "--catalog-csv",
            str(catalog),
            "--dataset",
            "fixture",
            "--processed-suffix",
            "tiny",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert code == 0
    assert (output_dir / "grounding_diagnostics_summary.json").exists()
    assert (output_dir / "grounding_diagnostics_report.md").exists()

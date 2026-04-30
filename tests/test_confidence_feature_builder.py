from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

from scripts.build_confidence_features import main as feature_cli_main
from storyflow.confidence import (
    CatalogFeatureIndex,
    build_confidence_features,
    confidence_feature_record,
    feature_from_grounded_row,
)
from storyflow.observation import read_jsonl, write_jsonl


def _workspace_tmp(name: str) -> Path:
    path = Path("outputs") / "test_tmp" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _catalog_csv(workspace: Path) -> Path:
    path = workspace / "item_catalog.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["item_id", "title", "popularity", "popularity_bucket"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"item_id": "item-head", "title": "Head Serum", "popularity": 100, "popularity_bucket": "head"},
                {"item_id": "item-mid", "title": "Mid Cleanser", "popularity": 30, "popularity_bucket": "mid"},
                {"item_id": "item-tail", "title": "Tail Balm", "popularity": 3, "popularity_bucket": "tail"},
            ]
        )
    return path


def _input_jsonl(workspace: Path) -> Path:
    path = workspace / "inputs.jsonl"
    write_jsonl(
        path,
        [
            {
                "input_id": "input-1",
                "history_item_ids": ["item-tail"],
                "catalog_candidate_item_ids": ["item-head", "item-mid"],
                "catalog_candidate_titles": ["Head Serum", "Mid Cleanser"],
                "catalog_candidate_popularity_buckets": ["head", "mid"],
            },
            {
                "input_id": "input-2",
                "history_item_ids": ["item-tail"],
            },
            {
                "input_id": "input-3",
                "history_item_ids": ["item-mid"],
            },
        ],
    )
    return path


def _grounded_jsonl(workspace: Path) -> Path:
    path = workspace / "grounded_predictions.jsonl"
    write_jsonl(
        path,
        [
            {
                "input_id": "input-1",
                "example_id": "ex-1",
                "user_id": "u1",
                "split": "test",
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "generated_title": "Head Serum",
                "confidence": 0.92,
                "target_item_id": "item-tail",
                "target_title": "Tail Balm",
                "target_popularity": 3,
                "target_popularity_bucket": "tail",
                "grounded_item_id": "item-head",
                "grounding_status": "exact",
                "grounding_score": 0.97,
                "grounding_ambiguity": 0.02,
                "correctness": 0,
                "is_experiment_result": False,
            },
            {
                "input_id": "input-2",
                "example_id": "ex-2",
                "user_id": "u2",
                "split": "test",
                "provider": "baseline",
                "baseline": "ranking_jsonl",
                "model": "ranking_jsonl",
                "generated_title": "Tail Balm",
                "confidence": 0.70,
                "baseline_score": 1.5,
                "baseline_score_source": "ranking_jsonl_sigmoid_score",
                "baseline_selected_rank": 2,
                "target_item_id": "item-tail",
                "target_title": "Tail Balm",
                "target_popularity": 3,
                "target_popularity_bucket": "tail",
                "grounded_item_id": "item-tail",
                "grounding_status": "exact",
                "grounding_score": 0.95,
                "grounding_ambiguity": 0.03,
                "correctness": 1,
                "is_experiment_result": False,
            },
            {
                "input_id": "input-3",
                "example_id": "ex-3",
                "user_id": "u3",
                "split": "test",
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "generated_title": "Imaginary Cream",
                "confidence": 0.81,
                "target_item_id": "item-mid",
                "target_title": "Mid Cleanser",
                "target_popularity": 30,
                "target_popularity_bucket": "mid",
                "grounded_item_id": None,
                "grounding_status": "ungrounded",
                "grounding_score": 0.0,
                "grounding_ambiguity": 1.0,
                "correctness": 0,
                "is_experiment_result": False,
            },
        ],
    )
    return path


def test_feature_from_grounded_row_uses_generated_catalog_popularity_not_target() -> None:
    workspace = _workspace_tmp("feature_one")
    catalog = CatalogFeatureIndex.from_csv(_catalog_csv(workspace))
    input_record = read_jsonl(_input_jsonl(workspace))[0]
    row = read_jsonl(_grounded_jsonl(workspace))[0]

    features, metadata = feature_from_grounded_row(row, catalog=catalog, input_record=input_record)

    assert features.item_id == "item-head"
    assert features.popularity_bucket == "head"
    assert features.popularity_percentile == 1.0
    assert features.verbal_confidence == 0.92
    assert features.grounding_confidence == 0.97
    assert features.correctness_label == 0
    assert features.history_alignment == 0.0
    assert features.novelty_score == 1.0
    assert metadata["target_popularity_bucket"] == "tail"
    assert metadata["popularity_source"] == "catalog_grounded_item"


def test_feature_builder_does_not_substitute_target_popularity_for_wrong_unknown_item() -> None:
    row = {
        "input_id": "x",
        "example_id": "ex",
        "user_id": "u",
        "generated_title": "Wrong Unknown",
        "confidence": 0.8,
        "target_item_id": "target-tail",
        "target_popularity_bucket": "tail",
        "grounded_item_id": "wrong-head",
        "grounding_status": "exact",
        "grounding_score": 0.8,
        "grounding_ambiguity": 0.1,
        "correctness": 0,
    }

    features, metadata = feature_from_grounded_row(row, catalog=None, input_record=None)

    assert features.popularity_bucket is None
    assert features.popularity_percentile is None
    assert metadata["popularity_source"] == "unknown"


def test_confidence_feature_record_includes_scaffold_score_and_abstains_ungrounded() -> None:
    workspace = _workspace_tmp("feature_record")
    catalog = CatalogFeatureIndex.from_csv(_catalog_csv(workspace))
    input_record = read_jsonl(_input_jsonl(workspace))[2]
    row = read_jsonl(_grounded_jsonl(workspace))[2]

    record = confidence_feature_record(row, catalog=catalog, input_record=input_record)

    assert record["feature_schema_version"] == "cure_truce_feature_v1"
    assert record["feature"]["is_grounded"] is False
    assert record["score"]["action"] == "abstain"
    assert record["score"]["risk_penalty"] == 1.0
    assert record["is_experiment_result"] is False


def test_build_confidence_features_writes_jsonl_and_manifest() -> None:
    workspace = _workspace_tmp("feature_build")
    catalog_csv = _catalog_csv(workspace)
    input_jsonl = _input_jsonl(workspace)
    grounded_jsonl = _grounded_jsonl(workspace)
    output_jsonl = workspace / "features.jsonl"
    manifest_json = workspace / "manifest.json"

    manifest = build_confidence_features(
        grounded_jsonl=grounded_jsonl,
        input_jsonl=input_jsonl,
        catalog_csv=catalog_csv,
        output_jsonl=output_jsonl,
        manifest_json=manifest_json,
    )
    rows = read_jsonl(output_jsonl)
    stored_manifest = json.loads(manifest_json.read_text(encoding="utf-8"))

    assert manifest["feature_count"] == 3
    assert stored_manifest["feature_schema_version"] == "cure_truce_feature_v1"
    assert stored_manifest["grounded_feature_count"] == 2
    assert stored_manifest["ungrounded_feature_count"] == 1
    assert stored_manifest["api_called"] is False
    assert rows[0]["feature"]["metadata"]["generated_popularity_bucket"] == "head"
    assert rows[1]["feature"]["preference_score"] > 0.5


def test_confidence_feature_cli_writes_outputs_without_api(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    workspace = _workspace_tmp("feature_cli")
    catalog_csv = _catalog_csv(workspace)
    input_jsonl = _input_jsonl(workspace)
    grounded_jsonl = _grounded_jsonl(workspace)
    output_dir = workspace / "features_cli"

    code = feature_cli_main(
        [
            "--grounded-jsonl",
            str(grounded_jsonl),
            "--input-jsonl",
            str(input_jsonl),
            "--catalog-csv",
            str(catalog_csv),
            "--output-dir",
            str(output_dir),
            "--max-examples",
            "2",
        ]
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert code == 0
    assert manifest["feature_count"] == 2
    assert manifest["api_called"] is False
    assert manifest["model_training"] is False
    assert (output_dir / "features.jsonl").exists()

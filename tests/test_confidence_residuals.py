from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from scripts.residualize_confidence_features import main as residual_cli_main
from storyflow.confidence import (
    fit_popularity_residual_model,
    residualize_feature_rows,
    row_popularity_bucket,
    row_popularity_percentile,
)
from storyflow.observation import read_jsonl, write_jsonl


def _workspace_tmp(name: str) -> Path:
    path = Path("outputs") / "test_tmp" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _feature_row(
    input_id: str,
    *,
    split: str,
    probability: float | None,
    bucket: str | None,
    label: int | None = None,
    percentile: float | None = None,
) -> dict[str, object]:
    feature: dict[str, object] = {
        "user_id": f"user-{input_id}",
        "item_id": f"item-{input_id}",
        "generated_title": f"Item {input_id}",
        "preference_score": 0.5,
        "verbal_confidence": None,
        "generation_confidence": None,
        "grounding_confidence": 0.9,
        "grounding_ambiguity": 0.05,
        "popularity_percentile": percentile,
        "popularity_bucket": bucket,
        "history_alignment": 0.2,
        "novelty_score": 0.8,
        "correctness_label": label,
        "is_grounded": True,
        "metadata": {"split": split, "generated_popularity_bucket": bucket},
    }
    score: dict[str, object] = {
        "item_id": f"item-{input_id}",
        "score": probability if probability is not None else 0.0,
        "estimated_exposure_confidence": probability,
        "risk_penalty": 0.0,
        "echo_risk": 0.0,
        "information_gain": 0.2,
        "popularity_residual": 0.0,
        "action": "recommend",
        "components": {},
    }
    return {
        "feature_schema_version": "cure_truce_feature_v1",
        "input_id": input_id,
        "example_id": f"ex-{input_id}",
        "user_id": f"user-{input_id}",
        "split": split,
        "feature": feature,
        "score": score,
        "is_experiment_result": False,
    }


def _residual_rows() -> list[dict[str, object]]:
    return [
        _feature_row("train-head-a", split="train", probability=0.90, bucket="head"),
        _feature_row("train-head-b", split="train", probability=0.80, bucket="head"),
        _feature_row("train-tail-a", split="train", probability=0.20, bucket="tail"),
        _feature_row("train-tail-b", split="train", probability=0.10, bucket="tail"),
        _feature_row(
            "valid-head",
            split="validation",
            probability=0.95,
            bucket="head",
            label=1,
            percentile=0.95,
        ),
        _feature_row(
            "valid-tail",
            split="validation",
            probability=0.45,
            bucket="tail",
            label=1,
            percentile=0.05,
        ),
        _feature_row("test-mid", split="test", probability=0.50, bucket="mid", label=0),
    ]


def test_popularity_residual_model_uses_fit_split_only() -> None:
    rows = _residual_rows() + [
        _feature_row("valid-head-low", split="validation", probability=0.05, bucket="head"),
        _feature_row("valid-tail-high", split="validation", probability=0.99, bucket="tail"),
    ]

    model = fit_popularity_residual_model(rows, fit_splits=("train",))

    assert model.fit_count == 4
    assert model.global_mean_probability == pytest.approx(0.50)
    assert model.baseline_probability("head") == (pytest.approx(0.85), False)
    assert model.baseline_probability("tail") == (pytest.approx(0.15), False)


def test_popularity_bucket_and_percentile_readers_prefer_feature_values() -> None:
    row = _feature_row(
        "reader",
        split="validation",
        probability=0.5,
        bucket="head",
        percentile=0.97,
    )

    assert row_popularity_bucket(row) == "head"
    assert row_popularity_percentile(row) == 0.97


def test_popularity_bucket_reader_does_not_borrow_target_bucket() -> None:
    row = _feature_row(
        "no-target-leak",
        split="validation",
        probability=0.5,
        bucket=None,
    )
    metadata = row["metadata"] = {"target_popularity_bucket": "tail"}
    assert metadata["target_popularity_bucket"] == "tail"

    assert row_popularity_bucket(row) == "unknown"


def test_residualize_feature_rows_refuses_fit_eval_overlap() -> None:
    workspace = _workspace_tmp("residual_overlap")
    features_jsonl = workspace / "features.jsonl"
    write_jsonl(features_jsonl, _residual_rows())

    with pytest.raises(ValueError, match="overlap"):
        residualize_feature_rows(
            features_jsonl=features_jsonl,
            output_jsonl=workspace / "residualized.jsonl",
            manifest_json=workspace / "manifest.json",
            fit_splits=("train",),
            eval_splits=("train",),
        )


def test_residualize_feature_rows_writes_eval_rows_and_manifest() -> None:
    workspace = _workspace_tmp("residual_manifest")
    features_jsonl = workspace / "features.jsonl"
    output_jsonl = workspace / "popularity_residualized_features.jsonl"
    manifest_json = workspace / "manifest.json"
    write_jsonl(features_jsonl, _residual_rows())

    manifest = residualize_feature_rows(
        features_jsonl=features_jsonl,
        output_jsonl=output_jsonl,
        manifest_json=manifest_json,
        fit_splits=("train",),
        eval_splits=("validation", "test"),
    )
    rows = read_jsonl(output_jsonl)
    stored_manifest = json.loads(manifest_json.read_text(encoding="utf-8"))

    assert manifest["fit_summary"]["fit_row_count"] == 4
    assert manifest["eval_summary"]["eval_row_count"] == 3
    assert stored_manifest["leakage_guard"]["fit_eval_overlap"] == []
    assert stored_manifest["api_called"] is False
    assert stored_manifest["model_training"] is False
    assert stored_manifest["is_experiment_result"] is False
    assert len(rows) == 3
    assert rows[0]["popularity_residualization"]["source_probability"] == 0.95
    assert rows[0]["popularity_residualization"]["popularity_baseline_probability"] == pytest.approx(
        0.85
    )
    assert rows[0]["popularity_residualization"][
        "popularity_residual_confidence"
    ] == pytest.approx(0.10)
    assert rows[1]["popularity_residualization"][
        "popularity_residual_confidence"
    ] == pytest.approx(0.30)
    assert rows[2]["popularity_residualization"]["used_global_bucket_fallback"] is True
    assert rows[2]["is_experiment_result"] is False


def test_residualize_feature_rows_keeps_missing_probability_rows() -> None:
    workspace = _workspace_tmp("residual_missing_probability")
    features_jsonl = workspace / "features.jsonl"
    output_jsonl = workspace / "residualized.jsonl"
    manifest_json = workspace / "manifest.json"
    rows = _residual_rows() + [
        _feature_row("valid-missing-prob", split="validation", probability=None, bucket="tail")
    ]
    write_jsonl(features_jsonl, rows)

    manifest = residualize_feature_rows(
        features_jsonl=features_jsonl,
        output_jsonl=output_jsonl,
        manifest_json=manifest_json,
        fit_splits=("train",),
        eval_splits=("validation",),
    )
    output_rows = read_jsonl(output_jsonl)
    missing = output_rows[-1]["popularity_residualization"]

    assert manifest["eval_summary"]["missing_eval_probability_count"] == 1
    assert missing["status"] == "missing_probability"
    assert missing["popularity_residual_confidence"] is None


def test_residual_cli_writes_outputs_without_api(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    workspace = _workspace_tmp("residual_cli")
    features_jsonl = workspace / "features.jsonl"
    output_dir = workspace / "residuals"
    write_jsonl(features_jsonl, _residual_rows())

    code = residual_cli_main(
        [
            "--features-jsonl",
            str(features_jsonl),
            "--output-dir",
            str(output_dir),
            "--fit-splits",
            "train",
            "--eval-splits",
            "validation,test",
        ]
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert code == 0
    assert manifest["popularity_residual_schema_version"] == "cure_truce_popularity_residual_v1"
    assert manifest["api_called"] is False
    assert manifest["server_executed"] is False
    assert (output_dir / "popularity_residualized_features.jsonl").exists()

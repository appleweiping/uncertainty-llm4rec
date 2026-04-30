from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from scripts.calibrate_confidence_features import main as calibrator_cli_main
from storyflow.confidence import (
    calibrate_feature_rows,
    fit_histogram_calibrator,
    row_probability,
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
    probability: float,
    label: int | None,
    verbal_confidence: float | None = None,
) -> dict[str, object]:
    return {
        "feature_schema_version": "cure_truce_feature_v1",
        "input_id": input_id,
        "example_id": f"ex-{input_id}",
        "user_id": f"user-{input_id}",
        "split": split,
        "feature": {
            "user_id": f"user-{input_id}",
            "item_id": f"item-{input_id}",
            "generated_title": f"Item {input_id}",
            "preference_score": 0.5,
            "verbal_confidence": verbal_confidence,
            "generation_confidence": None,
            "grounding_confidence": 0.9,
            "grounding_ambiguity": 0.05,
            "popularity_percentile": 0.4,
            "popularity_bucket": "mid",
            "history_alignment": 0.2,
            "novelty_score": 0.8,
            "correctness_label": label,
            "is_grounded": True,
            "metadata": {"split": split},
        },
        "score": {
            "item_id": f"item-{input_id}",
            "score": probability,
            "estimated_exposure_confidence": probability,
            "risk_penalty": 0.0,
            "echo_risk": 0.0,
            "information_gain": 0.2,
            "popularity_residual": 0.0,
            "action": "recommend",
            "components": {},
        },
        "is_experiment_result": False,
    }


def _calibration_rows() -> list[dict[str, object]]:
    return [
        _feature_row("train-low-a", split="train", probability=0.10, label=0),
        _feature_row("train-low-b", split="train", probability=0.20, label=0),
        _feature_row("train-high-a", split="train", probability=0.80, label=1),
        _feature_row("train-high-b", split="train", probability=0.90, label=1),
        _feature_row("valid-low", split="validation", probability=0.40, label=0),
        _feature_row("valid-high", split="validation", probability=0.60, label=1),
        _feature_row("test-high-wrong", split="test", probability=0.70, label=0),
    ]


def test_histogram_calibrator_uses_fit_split_only() -> None:
    rows = _calibration_rows() + [
        _feature_row("valid-high-wrong", split="validation", probability=0.95, label=0)
    ]

    calibrator = fit_histogram_calibrator(rows, fit_splits=("train",), n_bins=2)

    assert calibrator.fit_count == 4
    assert calibrator.bins[0].calibrated_probability == 0.0
    assert calibrator.bins[1].calibrated_probability == 1.0
    assert calibrator.calibrate(0.95) == (1.0, 1, False)


def test_calibration_reads_supported_probability_sources() -> None:
    row = _feature_row(
        "source-check",
        split="train",
        probability=0.33,
        label=1,
        verbal_confidence=0.77,
    )

    assert row_probability(row, source="estimated_exposure_confidence") == 0.33
    assert row_probability(row, source="verbal_confidence") == 0.77


def test_calibrate_feature_rows_refuses_fit_eval_overlap() -> None:
    workspace = _workspace_tmp("calibration_overlap")
    features_jsonl = workspace / "features.jsonl"
    write_jsonl(features_jsonl, _calibration_rows())

    with pytest.raises(ValueError, match="overlap"):
        calibrate_feature_rows(
            features_jsonl=features_jsonl,
            output_jsonl=workspace / "calibrated.jsonl",
            manifest_json=workspace / "manifest.json",
            fit_splits=("train",),
            eval_splits=("train",),
            n_bins=2,
        )


def test_calibrate_feature_rows_writes_eval_rows_and_manifest() -> None:
    workspace = _workspace_tmp("calibration_manifest")
    features_jsonl = workspace / "features.jsonl"
    output_jsonl = workspace / "calibrated_features.jsonl"
    manifest_json = workspace / "manifest.json"
    write_jsonl(features_jsonl, _calibration_rows())

    manifest = calibrate_feature_rows(
        features_jsonl=features_jsonl,
        output_jsonl=output_jsonl,
        manifest_json=manifest_json,
        fit_splits=("train",),
        eval_splits=("validation", "test"),
        n_bins=2,
    )
    rows = read_jsonl(output_jsonl)
    stored_manifest = json.loads(manifest_json.read_text(encoding="utf-8"))

    assert manifest["fit_summary"]["count"] == 4
    assert manifest["eval_summary"]["eval_row_count"] == 3
    assert stored_manifest["leakage_guard"]["fit_eval_overlap"] == []
    assert stored_manifest["api_called"] is False
    assert stored_manifest["model_training"] is False
    assert stored_manifest["is_experiment_result"] is False
    assert len(rows) == 3
    assert rows[0]["calibration"]["source_probability"] == 0.4
    assert rows[0]["calibration"]["calibrated_probability"] == 0.0
    assert rows[1]["calibration"]["calibrated_probability"] == 1.0
    assert rows[2]["is_experiment_result"] is False


def test_calibrate_feature_rows_requires_labeled_fit_rows() -> None:
    workspace = _workspace_tmp("calibration_missing_fit")
    features_jsonl = workspace / "features.jsonl"
    write_jsonl(
        features_jsonl,
        [
            _feature_row("train-unlabeled", split="train", probability=0.9, label=None),
            _feature_row("valid", split="validation", probability=0.8, label=1),
        ],
    )

    with pytest.raises(ValueError, match="no labeled fit rows"):
        calibrate_feature_rows(
            features_jsonl=features_jsonl,
            output_jsonl=workspace / "calibrated.jsonl",
            manifest_json=workspace / "manifest.json",
            fit_splits=("train",),
            eval_splits=("validation",),
            n_bins=2,
        )


def test_calibration_cli_writes_outputs_without_api(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    workspace = _workspace_tmp("calibration_cli")
    features_jsonl = workspace / "features.jsonl"
    output_dir = workspace / "calibration"
    write_jsonl(features_jsonl, _calibration_rows())

    code = calibrator_cli_main(
        [
            "--features-jsonl",
            str(features_jsonl),
            "--output-dir",
            str(output_dir),
            "--fit-splits",
            "train",
            "--eval-splits",
            "validation,test",
            "--n-bins",
            "2",
        ]
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert code == 0
    assert manifest["calibrator_schema_version"] == "cure_truce_calibrator_v1"
    assert manifest["api_called"] is False
    assert manifest["server_executed"] is False
    assert (output_dir / "calibrated_features.jsonl").exists()

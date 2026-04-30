from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from scripts.simulate_echo_exposure import main as simulate_cli_main
from scripts.triage_confidence_features import main as triage_cli_main
from storyflow.observation import read_jsonl, write_jsonl
from storyflow.simulation import (
    ExposureSimulationConfig,
    simulate_exposure_feedback_jsonl,
    simulate_exposure_feedback_rows,
)
from storyflow.triage import TriageConfig, triage_feature_rows, triage_features_jsonl


def _workspace_tmp(name: str) -> Path:
    path = Path("outputs") / "test_tmp" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _feature_row(
    input_id: str,
    item_id: str | None,
    *,
    title: str,
    split: str = "validation",
    preference_score: float = 0.6,
    score_confidence: float = 0.5,
    calibrated_probability: float | None = None,
    deconfounded_confidence: float | None = None,
    popularity_bucket: str | None = "mid",
    popularity_percentile: float | None = 0.5,
    history_alignment: float | None = 0.3,
    novelty_score: float | None = 0.7,
    grounding_confidence: float | None = 0.9,
    grounding_ambiguity: float | None = 0.05,
    verbal_confidence: float | None = 0.6,
    label: int | None = 1,
    grounded: bool = True,
) -> dict[str, object]:
    row: dict[str, object] = {
        "feature_schema_version": "cure_truce_feature_v1",
        "input_id": input_id,
        "example_id": f"ex-{input_id}-{title}",
        "user_id": f"user-{input_id}",
        "split": split,
        "feature": {
            "user_id": f"user-{input_id}",
            "item_id": item_id,
            "generated_title": title,
            "preference_score": preference_score,
            "verbal_confidence": verbal_confidence,
            "generation_confidence": None,
            "grounding_confidence": grounding_confidence,
            "grounding_ambiguity": grounding_ambiguity,
            "popularity_percentile": popularity_percentile,
            "popularity_bucket": popularity_bucket,
            "history_alignment": history_alignment,
            "novelty_score": novelty_score,
            "correctness_label": label,
            "is_grounded": grounded,
            "metadata": {
                "input_id": input_id,
                "split": split,
                "generated_popularity_bucket": popularity_bucket,
                "category": "beauty",
            },
        },
        "score": {
            "item_id": item_id,
            "score": score_confidence,
            "estimated_exposure_confidence": score_confidence,
            "risk_penalty": 0.0,
            "echo_risk": 0.0,
            "information_gain": 0.2,
            "popularity_residual": 0.0,
            "action": "recommend",
            "components": {},
        },
        "metadata": {"category": "beauty"},
        "is_experiment_result": False,
    }
    if calibrated_probability is not None:
        row["calibration"] = {
            "schema_version": "cure_truce_calibrator_v1",
            "calibrated_probability": calibrated_probability,
            "source_probability": score_confidence,
            "status": "calibrated",
        }
    if deconfounded_confidence is not None:
        row["popularity_residualization"] = {
            "schema_version": "cure_truce_popularity_residual_v1",
            "deconfounded_confidence_proxy": deconfounded_confidence,
            "popularity_residual_confidence": deconfounded_confidence - score_confidence,
            "status": "residualized",
        }
    return row


def _two_group_rows() -> list[dict[str, object]]:
    return [
        _feature_row(
            "g1",
            "head",
            title="Head Serum",
            preference_score=0.80,
            score_confidence=0.90,
            popularity_bucket="head",
            popularity_percentile=0.95,
            history_alignment=0.90,
            novelty_score=0.05,
            label=0,
        ),
        _feature_row(
            "g1",
            "tail",
            title="Tail Balm",
            preference_score=0.65,
            score_confidence=0.40,
            popularity_bucket="tail",
            popularity_percentile=0.05,
            history_alignment=0.10,
            novelty_score=0.90,
            label=1,
        ),
        _feature_row(
            "g2",
            "head",
            title="Head Serum",
            preference_score=0.75,
            score_confidence=0.88,
            popularity_bucket="head",
            popularity_percentile=0.95,
            history_alignment=0.85,
            novelty_score=0.05,
            label=0,
        ),
        _feature_row(
            "g2",
            "tail-2",
            title="Tail Oil",
            preference_score=0.70,
            score_confidence=0.45,
            popularity_bucket="tail",
            popularity_percentile=0.05,
            history_alignment=0.15,
            novelty_score=0.85,
            label=1,
        ),
    ]


def test_confidence_only_simulation_exposes_head_and_marks_synthetic() -> None:
    config = ExposureSimulationConfig(
        policies=("confidence_only",),
        rounds=1,
        confidence_source="score",
        feedback_learning_rate=0.5,
    )

    exposure_rows, summary = simulate_exposure_feedback_rows(_two_group_rows(), config=config)

    assert summary["synthetic_feedback"] is True
    assert summary["is_experiment_result"] is False
    assert len(exposure_rows) == 2
    assert {row["exposure_simulation"]["item_id"] for row in exposure_rows} == {"head"}
    final = summary["final_policy_summaries"]["confidence_only"]
    assert final["head_exposure_share"] == pytest.approx(1.0)
    assert final["tail_exposure_share"] == pytest.approx(0.0)
    assert final["exposure_gini"] > 0.0
    assert all(row["exposure_simulation"]["synthetic_feedback"] for row in exposure_rows)


def test_cure_truce_simulation_can_shift_to_tail_with_deconfounded_confidence() -> None:
    rows = [
        {
            **row,
            "popularity_residualization": {
                "schema_version": "cure_truce_popularity_residual_v1",
                "deconfounded_confidence_proxy": 0.35 if row["feature"]["item_id"] == "head" else 0.90,
                "status": "residualized",
            },
        }
        for row in _two_group_rows()
    ]
    config = ExposureSimulationConfig(
        policies=("cure_truce",),
        rounds=1,
        confidence_source="residualized",
    )

    exposure_rows, summary = simulate_exposure_feedback_rows(rows, config=config)

    assert {row["exposure_simulation"]["item_id"] for row in exposure_rows} == {
        "tail",
        "tail-2",
    }
    final = summary["final_policy_summaries"]["cure_truce"]
    assert final["tail_exposure_share"] == pytest.approx(1.0)
    assert final["head_exposure_share"] == pytest.approx(0.0)


def test_simulation_jsonl_and_cli_write_non_result_manifest() -> None:
    workspace = _workspace_tmp("echo_sim")
    features_jsonl = workspace / "features.jsonl"
    output_jsonl = workspace / "exposure_records.jsonl"
    summary_json = workspace / "summary.json"
    manifest_json = workspace / "manifest.json"
    write_jsonl(features_jsonl, _two_group_rows())

    manifest = simulate_exposure_feedback_jsonl(
        features_jsonl=features_jsonl,
        output_jsonl=output_jsonl,
        summary_json=summary_json,
        manifest_json=manifest_json,
        config=ExposureSimulationConfig(policies=("utility_only",), rounds=1),
    )

    rows = read_jsonl(output_jsonl)
    assert manifest["schema_version"] == "storyflow_echo_simulation_v1"
    assert manifest["api_called"] is False
    assert manifest["model_training"] is False
    assert manifest["server_executed"] is False
    assert manifest["is_experiment_result"] is False
    assert summary_json.exists()
    assert len(rows) == 2

    cli_dir = workspace / "cli"
    code = simulate_cli_main(
        [
            "--features-jsonl",
            str(features_jsonl),
            "--output-dir",
            str(cli_dir),
            "--policies",
            "confidence_only",
            "--rounds",
            "1",
            "--confidence-source",
            "score",
        ]
    )
    stored_manifest = json.loads((cli_dir / "manifest.json").read_text(encoding="utf-8"))
    assert code == 0
    assert stored_manifest["synthetic_feedback"] is True


def test_triage_preserves_underconfident_tail_positive() -> None:
    rows = [
        _feature_row(
            "g1",
            "tail",
            title="Tail Balm",
            score_confidence=0.20,
            popularity_bucket="tail",
            popularity_percentile=0.05,
            label=1,
        )
    ]

    output_rows, summary = triage_feature_rows(rows, config=TriageConfig(confidence_source="score"))

    triage = output_rows[0]["data_triage"]
    assert triage["action"] == "keep"
    assert triage["suggested_weight"] > 1.0
    assert "hard_tail_positive_underconfident" in triage["reason_codes"]
    assert summary["kept_hard_tail_positive_count"] == 1
    assert summary["is_experiment_result"] is False


def test_triage_downweights_wrong_high_confidence_head() -> None:
    rows = [
        _feature_row(
            "g1",
            "head",
            title="Head Serum",
            score_confidence=0.90,
            popularity_bucket="head",
            popularity_percentile=0.95,
            novelty_score=0.05,
            label=0,
        )
    ]

    output_rows, summary = triage_feature_rows(rows, config=TriageConfig(confidence_source="score"))

    triage = output_rows[0]["data_triage"]
    assert triage["action"] == "downweight"
    assert "wrong_high_confidence" in triage["reason_codes"]
    assert "popularity_or_echo_overconfident" in triage["reason_codes"]
    assert summary["downweight_ratio"] == pytest.approx(1.0)


def test_triage_jsonl_and_cli_write_non_result_manifest() -> None:
    workspace = _workspace_tmp("triage")
    features_jsonl = workspace / "features.jsonl"
    output_jsonl = workspace / "triaged_features.jsonl"
    manifest_json = workspace / "manifest.json"
    write_jsonl(features_jsonl, _two_group_rows())

    manifest = triage_features_jsonl(
        features_jsonl=features_jsonl,
        output_jsonl=output_jsonl,
        manifest_json=manifest_json,
        config=TriageConfig(confidence_source="score"),
    )

    rows = read_jsonl(output_jsonl)
    assert manifest["triage_schema_version"] == "storyflow_data_triage_v1"
    assert manifest["api_called"] is False
    assert manifest["model_training"] is False
    assert manifest["server_executed"] is False
    assert manifest["is_experiment_result"] is False
    assert manifest["selective_risk_diagnostics"]["overall"]["count"] == 4
    assert "head" in manifest["selective_risk_diagnostics"]["by_popularity_bucket"]
    assert len(rows) == 4
    assert "data_triage" in rows[0]

    cli_dir = workspace / "cli"
    code = triage_cli_main(
        [
            "--features-jsonl",
            str(features_jsonl),
            "--output-dir",
            str(cli_dir),
            "--confidence-source",
            "score",
        ]
    )
    stored_manifest = json.loads((cli_dir / "manifest.json").read_text(encoding="utf-8"))
    assert code == 0
    assert stored_manifest["triage_schema_version"] == "storyflow_data_triage_v1"

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from scripts.rerank_confidence_features import main as rerank_cli_main
from storyflow.confidence import (
    rerank_confidence_feature_rows,
    rerank_confidence_features_jsonl,
    select_rerank_confidence,
)
from storyflow.observation import read_jsonl, write_jsonl


def _workspace_tmp(name: str) -> Path:
    path = Path("outputs") / "test_tmp" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _feature_row(
    input_id: str,
    item_id: str,
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
) -> dict[str, object]:
    row: dict[str, object] = {
        "feature_schema_version": "cure_truce_feature_v1",
        "input_id": input_id,
        "example_id": f"ex-{input_id}",
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
            "is_grounded": True,
            "metadata": {"split": split, "generated_popularity_bucket": popularity_bucket},
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


def test_select_rerank_confidence_combines_calibrated_and_residualized() -> None:
    row = _feature_row(
        "g1",
        "item-a",
        title="A",
        calibrated_probability=0.7,
        deconfounded_confidence=0.5,
    )

    selected = select_rerank_confidence(row, confidence_source="calibrated_residualized")

    assert selected.value == pytest.approx(0.6)
    assert selected.selected_source == "calibrated_residualized"
    assert selected.fallback_used is False


def test_rerank_feature_rows_uses_residualized_confidence_and_echo_penalty() -> None:
    rows = [
        _feature_row(
            "group-1",
            "head",
            title="Head Serum",
            preference_score=0.85,
            score_confidence=0.80,
            deconfounded_confidence=0.40,
            popularity_bucket="head",
            popularity_percentile=0.95,
            history_alignment=0.90,
            novelty_score=0.05,
            grounding_confidence=0.95,
            verbal_confidence=0.90,
        ),
        _feature_row(
            "group-1",
            "tail",
            title="Tail Balm",
            preference_score=0.65,
            score_confidence=0.35,
            deconfounded_confidence=0.85,
            popularity_bucket="tail",
            popularity_percentile=0.05,
            history_alignment=0.20,
            novelty_score=0.85,
            grounding_confidence=0.90,
            verbal_confidence=0.60,
        ),
    ]

    output_rows, summary = rerank_confidence_feature_rows(
        rows,
        confidence_source="residualized",
        top_k=1,
    )

    assert summary["input_row_count"] == 2
    assert summary["output_row_count"] == 1
    assert summary["selected_confidence_source_counts"] == {"residualized": 1}
    assert output_rows[0]["cure_truce_rerank"]["item_id"] == "tail"
    assert output_rows[0]["cure_truce_rerank"]["rank"] == 1
    assert output_rows[0]["cure_truce_rerank"]["components"]["information_gain"] > 0.5
    assert output_rows[0]["is_experiment_result"] is False


def test_rerank_feature_rows_falls_back_to_score_when_calibration_missing() -> None:
    rows = [
        _feature_row("g1", "item-a", title="A", score_confidence=0.62),
    ]

    output_rows, summary = rerank_confidence_feature_rows(rows, confidence_source="calibrated")

    rerank = output_rows[0]["cure_truce_rerank"]
    assert rerank["selected_confidence_source"] == "score"
    assert rerank["selected_confidence"] == pytest.approx(0.62)
    assert rerank["fallback_used"] is True
    assert summary["fallback_count"] == 1


def test_rerank_feature_rows_strict_missing_confidence_raises() -> None:
    rows = [_feature_row("g1", "item-a", title="A", score_confidence=0.62)]

    with pytest.raises(ValueError, match="strict reranking"):
        rerank_confidence_feature_rows(
            rows,
            confidence_source="calibrated",
            strict_confidence_source=True,
        )


def test_rerank_confidence_features_jsonl_writes_manifest_without_api() -> None:
    workspace = _workspace_tmp("rerank_manifest")
    features_jsonl = workspace / "features.jsonl"
    output_jsonl = workspace / "reranked_features.jsonl"
    manifest_json = workspace / "manifest.json"
    write_jsonl(
        features_jsonl,
        [
            _feature_row(
                "g1",
                "item-a",
                title="A",
                calibrated_probability=0.70,
                deconfounded_confidence=0.80,
            ),
            _feature_row(
                "g1",
                "item-b",
                title="B",
                calibrated_probability=0.20,
                deconfounded_confidence=0.30,
            ),
            _feature_row("g2", "item-c", title="C", score_confidence=0.55),
        ],
    )

    manifest = rerank_confidence_features_jsonl(
        features_jsonl=features_jsonl,
        output_jsonl=output_jsonl,
        manifest_json=manifest_json,
        confidence_source="calibrated_residualized",
        top_k=1,
    )
    rows = read_jsonl(output_jsonl)
    stored_manifest = json.loads(manifest_json.read_text(encoding="utf-8"))

    assert manifest["reranker_schema_version"] == "cure_truce_reranker_v1"
    assert manifest["input_row_count"] == 3
    assert manifest["output_row_count"] == 2
    assert manifest["selective_risk_diagnostics"]["overall"]["count"] == 2
    assert manifest["selective_risk_diagnostics"]["api_called"] is False
    assert stored_manifest["api_called"] is False
    assert stored_manifest["model_training"] is False
    assert stored_manifest["server_executed"] is False
    assert stored_manifest["is_experiment_result"] is False
    assert len(rows) == 2
    assert rows[0]["cure_truce_rerank"]["rank"] == 1
    assert rows[1]["cure_truce_rerank"]["fallback_used"] is True


def test_rerank_cli_writes_outputs_without_api(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    workspace = _workspace_tmp("rerank_cli")
    features_jsonl = workspace / "features.jsonl"
    output_dir = workspace / "reranking"
    write_jsonl(
        features_jsonl,
        [
            _feature_row("g1", "item-a", title="A", score_confidence=0.50),
            _feature_row("g1", "item-b", title="B", score_confidence=0.60),
        ],
    )

    code = rerank_cli_main(
        [
            "--features-jsonl",
            str(features_jsonl),
            "--output-dir",
            str(output_dir),
            "--confidence-source",
            "score",
            "--top-k",
            "1",
        ]
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert code == 0
    assert manifest["reranker_schema_version"] == "cure_truce_reranker_v1"
    assert manifest["api_called"] is False
    assert manifest["server_executed"] is False
    assert (output_dir / "reranked_features.jsonl").exists()

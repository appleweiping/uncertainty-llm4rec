"""Shared evaluator for reproducible prediction artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.evaluation.export import export_metrics
from llm4rec.evaluation.prediction_schema import validate_prediction
from llm4rec.io.artifacts import read_jsonl
from llm4rec.metrics.calibration import calibration_metrics
from llm4rec.metrics.confidence import confidence_metrics
from llm4rec.metrics.efficiency import efficiency_metrics
from llm4rec.metrics.ranking import ranking_metrics
from llm4rec.metrics.validity import validity_metrics


def evaluate_predictions(
    *,
    predictions_jsonl: str | Path,
    output_dir: str | Path,
    top_k: list[int],
) -> dict[str, Any]:
    raw_rows = read_jsonl(predictions_jsonl)
    predictions = [
        validate_prediction(row, row_number=index + 1)
        for index, row in enumerate(raw_rows)
    ]
    aggregate = _compute(predictions, top_k=top_k)
    per_domain: dict[str, Any] = {}
    for domain in sorted({row["domain"] for row in predictions}):
        domain_rows = [row for row in predictions if row["domain"] == domain]
        per_domain[domain] = _compute(domain_rows, top_k=top_k)
    metrics = {
        "count": len(predictions),
        "top_k": top_k,
        "aggregate": aggregate,
        "per_domain": per_domain,
        "schema_version": "llm4rec_prediction_v1",
        "is_experiment_result": False,
        "note": _note_for_methods(predictions),
    }
    export_metrics(metrics, output_dir=output_dir)
    return metrics


def _compute(predictions: list[dict[str, Any]], *, top_k: list[int]) -> dict[str, Any]:
    ranking = ranking_metrics(predictions, top_k=top_k)
    validity = validity_metrics(predictions)
    return {
        **ranking,
        **validity,
        "confidence": confidence_metrics(predictions),
        "calibration": calibration_metrics(predictions),
        "efficiency": efficiency_metrics(predictions),
    }


def _note_for_methods(predictions: list[dict[str, Any]]) -> str:
    methods = {str(row.get("method") or "") for row in predictions}
    if methods == {"skeleton"}:
        return "Phase 1 skeleton smoke metrics only; not a formal baseline or paper result."
    if any(method.startswith("llm_") for method in methods):
        return (
            "Phase 3 mock LLM baseline and uncertainty-observation smoke metrics only; "
            "not a paper result or OursMethod claim."
        )
    return (
        "Phase 2 minimal baseline smoke metrics only; use for infrastructure "
        "validation, not as paper-level experimental evidence."
    )

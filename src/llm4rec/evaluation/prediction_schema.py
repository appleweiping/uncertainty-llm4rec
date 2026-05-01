"""Prediction JSONL schema validation."""

from __future__ import annotations

from typing import Any

REQUIRED_FIELDS = {
    "user_id",
    "target_item",
    "candidate_items",
    "predicted_items",
    "method",
    "domain",
    "metadata",
}

OURS_METADATA_FIELDS = {
    "ours_method",
    "generated_title",
    "confidence",
    "grounding_success",
    "grounded_item_id",
    "uncertainty_decision",
    "fallback_method",
    "echo_risk",
    "popularity_bucket",
    "history_similarity",
    "ablation_variant",
    "disabled_components",
    "parse_success",
    "prompt_template_id",
    "prompt_hash",
}

OURS_DECISIONS = {"accept", "fallback", "abstain", "rerank"}


def validate_prediction(record: dict[str, Any], *, row_number: int | None = None) -> dict[str, Any]:
    where = f"row {row_number}: " if row_number is not None else ""
    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        raise ValueError(f"{where}prediction missing required fields: {missing}")
    if not isinstance(record["user_id"], str) or not record["user_id"]:
        raise ValueError(f"{where}user_id must be a non-empty string")
    if not isinstance(record["target_item"], str) or not record["target_item"]:
        raise ValueError(f"{where}target_item must be a non-empty string")
    if not _is_str_list(record["candidate_items"]):
        raise ValueError(f"{where}candidate_items must be list[str]")
    if not _is_str_list(record["predicted_items"]):
        raise ValueError(f"{where}predicted_items must be list[str]")
    scores = record.get("scores")
    if scores in (None, []):
        scores = None
    elif not isinstance(scores, list):
        raise ValueError(f"{where}scores must be list[number], null, or empty")
    elif len(scores) != len(record["predicted_items"]):
        raise ValueError(f"{where}scores length must equal predicted_items length")
    else:
        for index, score in enumerate(scores):
            if not isinstance(score, (int, float)):
                raise ValueError(f"{where}scores[{index}] must be numeric")
    if not isinstance(record["method"], str) or not record["method"]:
        raise ValueError(f"{where}method must be a non-empty string")
    if not isinstance(record["domain"], str) or not record["domain"]:
        raise ValueError(f"{where}domain must be a non-empty string")
    if not isinstance(record["metadata"], dict):
        raise ValueError(f"{where}metadata must be an object")
    _validate_ours_metadata(record["metadata"], where=where)
    raw_output = record.get("raw_output")
    if raw_output is not None and not isinstance(raw_output, str):
        raise ValueError(f"{where}raw_output must be a string or null")
    normalized = dict(record)
    normalized["scores"] = scores
    return normalized


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _validate_ours_metadata(metadata: dict[str, Any], *, where: str) -> None:
    if metadata.get("ours_method") is not True:
        return
    missing = sorted(OURS_METADATA_FIELDS - set(metadata))
    if missing:
        raise ValueError(f"{where}OursMethod metadata missing required fields: {missing}")
    confidence = metadata.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"{where}OursMethod metadata confidence must be numeric")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError(f"{where}OursMethod metadata confidence must be in [0, 1]")
    if metadata.get("uncertainty_decision") not in OURS_DECISIONS:
        raise ValueError(f"{where}OursMethod uncertainty_decision is invalid")
    if not isinstance(metadata.get("grounding_success"), bool):
        raise ValueError(f"{where}OursMethod grounding_success must be bool")
    if not isinstance(metadata.get("echo_risk"), bool):
        raise ValueError(f"{where}OursMethod echo_risk must be bool")
    if not isinstance(metadata.get("parse_success"), bool):
        raise ValueError(f"{where}OursMethod parse_success must be bool")
    if not isinstance(metadata.get("disabled_components"), list):
        raise ValueError(f"{where}OursMethod disabled_components must be a list")
    if not isinstance(metadata.get("prompt_template_id"), str):
        raise ValueError(f"{where}OursMethod prompt_template_id must be a string")
    if not isinstance(metadata.get("prompt_hash"), str):
        raise ValueError(f"{where}OursMethod prompt_hash must be a string")

"""Uncertainty-aware data triage reason codes for CURE/TRUCE feature rows.

The triage scaffold consumes the same grounded feature JSONL used by
calibration, residualization, reranking, and exposure simulation. It assigns
diagnostic reason codes and suggested weights without deleting data or claiming
that a sample is truly noisy. This keeps noisy cases, hard tail positives, and
grounding uncertainty separate for later approved training work.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from storyflow.confidence import (
    DEFAULT_RERANK_CONFIDENCE_SOURCE,
    SUPPORTED_RERANK_CONFIDENCE_SOURCES,
    compute_echo_risk,
    feature_from_rerank_row,
    select_rerank_confidence,
)
from storyflow.observation import read_jsonl, utc_now_iso, write_jsonl

TRIAGE_SCHEMA_VERSION = "storyflow_data_triage_v1"


@dataclass(frozen=True, slots=True)
class TriageConfig:
    """Thresholds for conservative diagnostic triage."""

    confidence_source: str = DEFAULT_RERANK_CONFIDENCE_SOURCE
    high_confidence_threshold: float = 0.70
    low_confidence_threshold: float = 0.35
    low_grounding_threshold: float = 0.45
    high_ambiguity_threshold: float = 0.50
    high_echo_risk_threshold: float = 0.35
    low_novelty_threshold: float = 0.25

    def __post_init__(self) -> None:
        if self.confidence_source not in SUPPORTED_RERANK_CONFIDENCE_SOURCES:
            raise ValueError(f"unsupported confidence source: {self.confidence_source}")
        for field_name, value in asdict(self).items():
            if field_name == "confidence_source":
                continue
            if not 0 <= float(value) <= 1:
                raise ValueError(f"{field_name} must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _feature_metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    feature = row.get("feature")
    if isinstance(feature, Mapping):
        metadata = feature.get("metadata")
        if isinstance(metadata, Mapping):
            return metadata
    return {}


def _bucket(row: Mapping[str, Any]) -> str:
    feature = row.get("feature")
    if isinstance(feature, Mapping):
        bucket = feature.get("popularity_bucket")
        if bucket not in (None, ""):
            return str(bucket)
    metadata = _feature_metadata(row)
    bucket = metadata.get("generated_popularity_bucket")
    return str(bucket) if bucket not in (None, "") else "unknown"


def _novelty(row: Mapping[str, Any]) -> float | None:
    feature = row.get("feature")
    if not isinstance(feature, Mapping):
        return None
    value = feature.get("novelty_score")
    if value in (None, ""):
        return None
    return float(value)


def _grounding_confidence(row: Mapping[str, Any]) -> float | None:
    feature = row.get("feature")
    if not isinstance(feature, Mapping):
        return None
    value = feature.get("grounding_confidence")
    if value in (None, ""):
        return None
    return float(value)


def _grounding_ambiguity(row: Mapping[str, Any]) -> float | None:
    feature = row.get("feature")
    if not isinstance(feature, Mapping):
        return None
    value = feature.get("grounding_ambiguity")
    if value in (None, ""):
        return None
    return float(value)


def _label(row: Mapping[str, Any]) -> int | None:
    feature = row.get("feature")
    if not isinstance(feature, Mapping):
        return None
    value = feature.get("correctness_label")
    if value in (None, ""):
        return None
    label = int(value)
    if label not in (0, 1):
        raise ValueError("correctness_label must be 0, 1, or missing")
    return label


def _triage_decision(row: Mapping[str, Any], config: TriageConfig) -> dict[str, Any]:
    features = feature_from_rerank_row(row)
    selected = select_rerank_confidence(
        row,
        confidence_source=config.confidence_source,
        strict_confidence_source=False,
    )
    confidence = selected.value
    label = _label(row)
    bucket = _bucket(row)
    novelty = _novelty(row)
    grounding_confidence = _grounding_confidence(row)
    grounding_ambiguity = _grounding_ambiguity(row)
    echo_risk = compute_echo_risk(features, confidence)

    reason_codes: list[str] = []
    action = "keep"
    suggested_weight = 1.0

    grounding_uncertain = (
        not features.is_grounded
        or (
            grounding_confidence is not None
            and grounding_confidence < config.low_grounding_threshold
        )
        or (
            grounding_ambiguity is not None
            and grounding_ambiguity > config.high_ambiguity_threshold
        )
    )
    if grounding_uncertain:
        reason_codes.append("grounding_uncertain")
        action = "review"
        suggested_weight = 0.50

    if not features.is_grounded and confidence >= config.high_confidence_threshold:
        reason_codes.append("ungrounded_high_confidence_noise_candidate")
        action = "prune_candidate"
        suggested_weight = 0.0

    if label == 1 and bucket == "tail" and confidence <= config.low_confidence_threshold:
        reason_codes.append("hard_tail_positive_underconfident")
        action = "keep"
        suggested_weight = 1.25

    if label == 0 and confidence >= config.high_confidence_threshold:
        reason_codes.append("wrong_high_confidence")
        if bucket == "head" or echo_risk >= config.high_echo_risk_threshold:
            reason_codes.append("popularity_or_echo_overconfident")
            action = "downweight"
            suggested_weight = min(suggested_weight, 0.25)
        elif action == "keep":
            action = "review"
            suggested_weight = min(suggested_weight, 0.50)

    if (
        bucket == "head"
        and confidence >= config.high_confidence_threshold
        and novelty is not None
        and novelty <= config.low_novelty_threshold
    ):
        reason_codes.append("head_low_novelty_overconfident")
        if action == "keep":
            action = "downweight"
            suggested_weight = 0.50

    if label is None:
        reason_codes.append("missing_correctness_label")
        if action == "keep":
            action = "review"
            suggested_weight = 0.75

    if not reason_codes:
        reason_codes.append("ordinary_keep")

    return {
        "schema_version": TRIAGE_SCHEMA_VERSION,
        "action": action,
        "reason_codes": reason_codes,
        "suggested_weight": suggested_weight,
        "selected_confidence": confidence,
        "selected_confidence_source": selected.selected_source,
        "requested_confidence_source": selected.requested_source,
        "confidence_fallback_used": selected.fallback_used,
        "correctness_label": label,
        "popularity_bucket": bucket,
        "novelty_score": novelty,
        "grounding_confidence": grounding_confidence,
        "grounding_ambiguity": grounding_ambiguity,
        "echo_risk": echo_risk,
        "synthetic_or_scaffold_triage": True,
        "api_called": False,
        "model_training": False,
        "server_executed": False,
        "is_experiment_result": False,
    }


def triage_feature_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: TriageConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign diagnostic triage reason codes to feature rows."""

    config = config or TriageConfig()
    input_rows = [dict(row) for row in rows]
    if not input_rows:
        raise ValueError("cannot triage an empty feature row collection")

    output_rows: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    for row in input_rows:
        triage = _triage_decision(row, config)
        output = dict(row)
        output["is_experiment_result"] = False
        output["data_triage"] = triage
        output_rows.append(output)
        action_counts[triage["action"]] += 1
        bucket_counts[str(triage["popularity_bucket"])] += 1
        for reason in triage["reason_codes"]:
            reason_counts[str(reason)] += 1

    total = len(output_rows)
    hard_tail_positive_count = reason_counts.get("hard_tail_positive_underconfident", 0)
    summary = {
        "triage_schema_version": TRIAGE_SCHEMA_VERSION,
        "config": config.to_dict(),
        "input_row_count": total,
        "output_row_count": total,
        "action_counts": dict(sorted(action_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "popularity_bucket_counts": dict(sorted(bucket_counts.items())),
        "prune_ratio": action_counts.get("prune_candidate", 0) / total,
        "downweight_ratio": action_counts.get("downweight", 0) / total,
        "review_ratio": action_counts.get("review", 0) / total,
        "hard_tail_positive_count": hard_tail_positive_count,
        "kept_hard_tail_positive_count": sum(
            1
            for row in output_rows
            if "hard_tail_positive_underconfident" in row["data_triage"]["reason_codes"]
            and row["data_triage"]["action"] == "keep"
        ),
        "api_called": False,
        "model_training": False,
        "server_executed": False,
        "is_experiment_result": False,
    }
    return output_rows, summary


def triage_features_jsonl(
    *,
    features_jsonl: str | Path,
    output_jsonl: str | Path,
    manifest_json: str | Path,
    config: TriageConfig | None = None,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Triage feature JSONL rows and write output plus a manifest."""

    features_path = Path(features_jsonl)
    rows = read_jsonl(features_path)
    if max_examples is not None:
        if max_examples < 1:
            raise ValueError("max_examples must be >= 1")
        rows = rows[:max_examples]
    output_rows, summary = triage_feature_rows(rows, config=config)
    write_jsonl(output_jsonl, output_rows)
    manifest = {
        "created_at_utc": utc_now_iso(),
        **summary,
        "features_jsonl": str(features_path),
        "output_jsonl": str(output_jsonl),
        "manifest_json": str(manifest_json),
        "max_examples": max_examples,
        "row_contract": {
            "input_requires": "CURE/TRUCE feature rows",
            "output_adds": "data_triage",
            "triage_is_diagnostic": True,
            "do_not_delete_hard_tail_positive": True,
        },
        "note": (
            "Diagnostic data triage scaffold. Reason codes and suggested "
            "weights separate likely-noise candidates from hard tail positives; "
            "this does not train a model or prove a pruning policy."
        ),
    }
    manifest_path = Path(manifest_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "TRIAGE_SCHEMA_VERSION",
    "TriageConfig",
    "triage_feature_rows",
    "triage_features_jsonl",
]

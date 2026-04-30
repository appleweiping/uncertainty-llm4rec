"""CURE/TRUCE reranking contract for calibrated/residualized feature rows.

This module connects the Phase 4 feature, calibration, and popularity-residual
scaffolds into one deterministic JSONL reranker. It is a contract for later
learned rerankers; it does not call APIs, train models, or produce paper
evidence.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

from storyflow.confidence.exposure import (
    CureTruceScore,
    CureTruceWeights,
    ExposureConfidenceFeatures,
    compute_echo_risk,
    compute_information_gain,
    compute_risk_penalty,
    popularity_pressure,
)
from storyflow.confidence.calibration import row_split
from storyflow.observation import read_jsonl, utc_now_iso, write_jsonl

RERANKER_SCHEMA_VERSION = "cure_truce_reranker_v1"
DEFAULT_RERANK_CONFIDENCE_SOURCE = "calibrated_residualized"
SUPPORTED_RERANK_CONFIDENCE_SOURCES = (
    "score",
    "calibrated",
    "residualized",
    "calibrated_residualized",
)
_VALID_POPULARITY_BUCKETS = {"head", "mid", "tail"}


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return _clip01(numeric)


def _group_id(row: Mapping[str, Any], *, group_key: str) -> str:
    value = row.get(group_key)
    if value in (None, ""):
        feature = row.get("feature")
        if isinstance(feature, Mapping):
            metadata = feature.get("metadata")
            if isinstance(metadata, Mapping):
                value = metadata.get(group_key)
    if value in (None, ""):
        raise ValueError(f"feature row is missing group key: {group_key}")
    return str(value)


def _safe_split(row: Mapping[str, Any]) -> str:
    try:
        return row_split(row)
    except ValueError:
        return "unknown"


def _confidence_sources(row: Mapping[str, Any]) -> dict[str, float | None]:
    score = row.get("score")
    calibration = row.get("calibration")
    residualization = row.get("popularity_residualization")
    return {
        "score": _float_or_none(
            score.get("estimated_exposure_confidence") if isinstance(score, Mapping) else None
        ),
        "calibrated": _float_or_none(
            calibration.get("calibrated_probability")
            if isinstance(calibration, Mapping)
            else None
        ),
        "residualized": _float_or_none(
            residualization.get("deconfounded_confidence_proxy")
            if isinstance(residualization, Mapping)
            else None
        ),
    }


@dataclass(frozen=True, slots=True)
class SelectedRerankConfidence:
    """Confidence value selected for one reranking row."""

    requested_source: str
    selected_source: str
    value: float
    available_sources: Mapping[str, float]
    fallback_used: bool
    fallback_reason: str | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_rerank_confidence(
    row: Mapping[str, Any],
    *,
    confidence_source: str = DEFAULT_RERANK_CONFIDENCE_SOURCE,
    strict_confidence_source: bool = False,
) -> SelectedRerankConfidence:
    """Select the confidence proxy used by the row-level reranker."""

    if confidence_source not in SUPPORTED_RERANK_CONFIDENCE_SOURCES:
        raise ValueError(f"unsupported rerank confidence source: {confidence_source}")

    sources = _confidence_sources(row)
    available = {key: value for key, value in sources.items() if value is not None}

    if confidence_source == "calibrated_residualized":
        calibrated = sources["calibrated"]
        residualized = sources["residualized"]
        if calibrated is not None and residualized is not None:
            return SelectedRerankConfidence(
                requested_source=confidence_source,
                selected_source=confidence_source,
                value=_clip01((calibrated + residualized) / 2.0),
                available_sources=available,
                fallback_used=False,
                fallback_reason=None,
                status="selected",
            )
        if strict_confidence_source:
            raise ValueError(
                "strict calibrated_residualized reranking requires both "
                "calibration.calibrated_probability and "
                "popularity_residualization.deconfounded_confidence_proxy"
            )
        for fallback in ("residualized", "calibrated", "score"):
            value = sources[fallback]
            if value is not None:
                return SelectedRerankConfidence(
                    requested_source=confidence_source,
                    selected_source=fallback,
                    value=value,
                    available_sources=available,
                    fallback_used=True,
                    fallback_reason=f"missing_{confidence_source}",
                    status="fallback",
                )
    else:
        requested_value = sources[confidence_source]
        if requested_value is not None:
            return SelectedRerankConfidence(
                requested_source=confidence_source,
                selected_source=confidence_source,
                value=requested_value,
                available_sources=available,
                fallback_used=False,
                fallback_reason=None,
                status="selected",
            )
        if strict_confidence_source:
            raise ValueError(f"strict reranking requested missing source: {confidence_source}")
        score_value = sources["score"]
        if score_value is not None:
            return SelectedRerankConfidence(
                requested_source=confidence_source,
                selected_source="score",
                value=score_value,
                available_sources=available,
                fallback_used=True,
                fallback_reason=f"missing_{confidence_source}",
                status="fallback",
            )

    return SelectedRerankConfidence(
        requested_source=confidence_source,
        selected_source="zero_fallback",
        value=0.0,
        available_sources=available,
        fallback_used=True,
        fallback_reason="missing_all_confidence_sources",
        status="missing_confidence_fallback_zero",
    )


def feature_from_rerank_row(row: Mapping[str, Any]) -> ExposureConfidenceFeatures:
    """Reconstruct ``ExposureConfidenceFeatures`` from a feature JSONL row."""

    feature = row.get("feature")
    if not isinstance(feature, Mapping):
        raise ValueError("reranking requires rows with a feature object")

    allowed_fields = {field.name for field in fields(ExposureConfidenceFeatures)}
    raw: dict[str, Any] = {
        key: feature.get(key) for key in allowed_fields if key in feature
    }
    if raw.get("user_id") in (None, "") and row.get("user_id") not in (None, ""):
        raw["user_id"] = row.get("user_id")
    if raw.get("generated_title") in (None, "") and row.get("generated_title") not in (None, ""):
        raw["generated_title"] = row.get("generated_title")
    if raw.get("item_id") == "":
        raw["item_id"] = None
    if str(raw.get("popularity_bucket") or "").strip().lower() not in _VALID_POPULARITY_BUCKETS:
        raw["popularity_bucket"] = None
    if not isinstance(raw.get("metadata"), Mapping):
        raw["metadata"] = {}
    if raw.get("is_grounded") is None:
        raw["is_grounded"] = bool(raw.get("item_id"))
    return ExposureConfidenceFeatures(**raw)


def _choose_action(
    *,
    features: ExposureConfidenceFeatures,
    selected_confidence: float,
    risk_penalty: float,
    echo_risk: float,
    information_gain: float,
    weights: CureTruceWeights,
) -> str:
    if not features.is_grounded:
        return "abstain"
    if risk_penalty >= weights.abstain_risk_threshold:
        return "abstain"
    if echo_risk >= weights.diversify_echo_threshold:
        return "diversify"
    if selected_confidence >= weights.recommend_confidence_threshold:
        return "recommend"
    if information_gain >= weights.explore_information_threshold:
        return "explore"
    return "abstain"


def score_rerank_feature_row(
    row: Mapping[str, Any],
    *,
    confidence_source: str = DEFAULT_RERANK_CONFIDENCE_SOURCE,
    strict_confidence_source: bool = False,
    weights: CureTruceWeights | None = None,
) -> tuple[ExposureConfidenceFeatures, CureTruceScore, SelectedRerankConfidence]:
    """Score one serialized feature row using the selected confidence source."""

    weights = weights or CureTruceWeights()
    features = feature_from_rerank_row(row)
    selected = select_rerank_confidence(
        row,
        confidence_source=confidence_source,
        strict_confidence_source=strict_confidence_source,
    )
    risk_penalty = compute_risk_penalty(features, selected.value, weights)
    echo_risk = compute_echo_risk(features, selected.value)
    information_gain = compute_information_gain(features)
    popularity_residual = selected.value - popularity_pressure(features)
    score = (
        weights.preference_score_weight * features.preference_score
        + weights.exposure_confidence_weight * selected.value
        + weights.information_gain_weight * information_gain
        - weights.risk_penalty_weight * risk_penalty
        - weights.echo_penalty_weight * echo_risk
    )
    action = _choose_action(
        features=features,
        selected_confidence=selected.value,
        risk_penalty=risk_penalty,
        echo_risk=echo_risk,
        information_gain=information_gain,
        weights=weights,
    )
    return (
        features,
        CureTruceScore(
            item_id=features.item_id,
            score=score,
            estimated_exposure_confidence=selected.value,
            risk_penalty=risk_penalty,
            echo_risk=echo_risk,
            information_gain=information_gain,
            popularity_residual=popularity_residual,
            action=action,
            components={
                "preference_score": features.preference_score,
                "selected_confidence": selected.value,
                "selected_confidence_source": selected.selected_source,
                "requested_confidence_source": selected.requested_source,
                "available_confidence_sources": dict(selected.available_sources),
                "popularity_pressure": popularity_pressure(features),
                "popularity_bucket": features.popularity_bucket,
                "grounding_confidence": features.grounding_confidence,
                "grounding_ambiguity": features.grounding_ambiguity,
                "history_alignment": features.history_alignment,
                "novelty_score": features.novelty_score,
                "correctness_label": features.correctness_label,
                "is_grounded": features.is_grounded,
            },
        ),
        selected,
    )


def _group_size_summary(sizes: Iterable[int]) -> dict[str, float | int | None]:
    values = list(sizes)
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def rerank_confidence_feature_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    confidence_source: str = DEFAULT_RERANK_CONFIDENCE_SOURCE,
    group_key: str = "input_id",
    top_k: int | None = None,
    strict_confidence_source: bool = False,
    weights: CureTruceWeights | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rerank serialized CURE/TRUCE feature rows by group."""

    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1 when provided")
    input_rows = [dict(row) for row in rows]
    if not input_rows:
        raise ValueError("cannot rerank an empty feature row collection")

    weights = weights or CureTruceWeights()
    grouped: dict[str, list[dict[str, Any]]] = {}
    group_order: list[str] = []
    scored_count = 0
    for row_index, row in enumerate(input_rows):
        group_id = _group_id(row, group_key=group_key)
        if group_id not in grouped:
            grouped[group_id] = []
            group_order.append(group_id)
        features, score, selected = score_rerank_feature_row(
            row,
            confidence_source=confidence_source,
            strict_confidence_source=strict_confidence_source,
            weights=weights,
        )
        output = dict(row)
        output["is_experiment_result"] = False
        output["cure_truce_rerank"] = {
            "schema_version": RERANKER_SCHEMA_VERSION,
            "requested_confidence_source": confidence_source,
            "selected_confidence_source": selected.selected_source,
            "selected_confidence": selected.value,
            "available_confidence_sources": dict(selected.available_sources),
            "fallback_used": selected.fallback_used,
            "fallback_reason": selected.fallback_reason,
            "status": selected.status,
            "group_key": group_key,
            "group_id": group_id,
            "input_row_index": row_index,
            "item_id": features.item_id,
            "generated_title": features.generated_title,
            "score": score.score,
            "action": score.action,
            "components": score.to_dict(),
            "api_called": False,
            "model_training": False,
            "server_executed": False,
            "is_experiment_result": False,
        }
        grouped[group_id].append(output)
        scored_count += 1

    output_rows: list[dict[str, Any]] = []
    group_sizes: list[int] = []
    for group_id in group_order:
        group_rows = grouped[group_id]
        group_sizes.append(len(group_rows))
        group_rows.sort(
            key=lambda row: (
                -float(row["cure_truce_rerank"]["score"]),
                str(row["cure_truce_rerank"].get("item_id") or ""),
                str(row["cure_truce_rerank"].get("generated_title") or ""),
                int(row["cure_truce_rerank"]["input_row_index"]),
            )
        )
        kept_rows = group_rows[:top_k] if top_k is not None else group_rows
        for rank, row in enumerate(kept_rows, start=1):
            row["cure_truce_rerank"]["rank"] = rank
            row["cure_truce_rerank"]["group_size_before_top_k"] = len(group_rows)
            row["cure_truce_rerank"]["top_k"] = top_k
            output_rows.append(row)

    action_counts = Counter(row["cure_truce_rerank"]["action"] for row in output_rows)
    source_counts = Counter(
        row["cure_truce_rerank"]["selected_confidence_source"] for row in output_rows
    )
    split_counts = Counter(_safe_split(row) for row in output_rows)
    fallback_count = sum(bool(row["cure_truce_rerank"]["fallback_used"]) for row in output_rows)
    zero_fallback_count = sum(
        row["cure_truce_rerank"]["selected_confidence_source"] == "zero_fallback"
        for row in output_rows
    )
    summary = {
        "reranker_schema_version": RERANKER_SCHEMA_VERSION,
        "requested_confidence_source": confidence_source,
        "group_key": group_key,
        "top_k": top_k,
        "strict_confidence_source": strict_confidence_source,
        "input_row_count": len(input_rows),
        "scored_row_count": scored_count,
        "output_row_count": len(output_rows),
        "group_count": len(grouped),
        "group_size_summary": _group_size_summary(group_sizes),
        "split_counts": dict(sorted(split_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "selected_confidence_source_counts": dict(sorted(source_counts.items())),
        "fallback_count": fallback_count,
        "zero_confidence_fallback_count": zero_fallback_count,
        "api_called": False,
        "model_training": False,
        "server_executed": False,
        "is_experiment_result": False,
    }
    return output_rows, summary


def rerank_confidence_features_jsonl(
    *,
    features_jsonl: str | Path,
    output_jsonl: str | Path,
    manifest_json: str | Path,
    confidence_source: str = DEFAULT_RERANK_CONFIDENCE_SOURCE,
    group_key: str = "input_id",
    top_k: int | None = None,
    strict_confidence_source: bool = False,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Rerank feature-row JSONL and write reranked rows plus a manifest."""

    features_path = Path(features_jsonl)
    rows = read_jsonl(features_path)
    if max_examples is not None:
        if max_examples < 1:
            raise ValueError("max_examples must be >= 1")
        rows = rows[:max_examples]
    reranked_rows, summary = rerank_confidence_feature_rows(
        rows,
        confidence_source=confidence_source,
        group_key=group_key,
        top_k=top_k,
        strict_confidence_source=strict_confidence_source,
    )
    write_jsonl(output_jsonl, reranked_rows)
    manifest = {
        "created_at_utc": utc_now_iso(),
        **summary,
        "features_jsonl": str(features_path),
        "output_jsonl": str(output_jsonl),
        "manifest_json": str(manifest_json),
        "max_examples": max_examples,
        "row_contract": {
            "input_requires": "CURE/TRUCE feature rows with a feature object",
            "output_adds": "cure_truce_rerank",
            "default_group_key": "input_id",
            "grounding_required_before_evaluation": True,
            "generated_title_must_be_grounded_before_correctness": True,
        },
        "note": (
            "Deterministic CURE/TRUCE reranking scaffold over existing feature, "
            "calibration, or popularity-residual rows. This is a JSONL contract "
            "for later learned rerankers; it is not a trained method result or "
            "paper evidence."
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
    "DEFAULT_RERANK_CONFIDENCE_SOURCE",
    "RERANKER_SCHEMA_VERSION",
    "SUPPORTED_RERANK_CONFIDENCE_SOURCES",
    "SelectedRerankConfidence",
    "feature_from_rerank_row",
    "rerank_confidence_feature_rows",
    "rerank_confidence_features_jsonl",
    "score_rerank_feature_row",
    "select_rerank_confidence",
]

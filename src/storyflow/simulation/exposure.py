"""Synthetic confidence-guided exposure simulation.

This module is the first Phase 5 scaffold. It consumes the existing
CURE/TRUCE feature-row JSONL contract and simulates how different exposure
policies would allocate impressions when confidence is allowed to influence
selection. The feedback update is synthetic and diagnostic only; it is not a
user-behavior model, not a server run, and not paper evidence.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from storyflow.confidence import (
    DEFAULT_RERANK_CONFIDENCE_SOURCE,
    SUPPORTED_RERANK_CONFIDENCE_SOURCES,
    CureTruceWeights,
    ExposureConfidenceFeatures,
    compute_echo_risk,
    compute_information_gain,
    compute_risk_penalty,
    feature_from_rerank_row,
    popularity_pressure,
    select_rerank_confidence,
)
from storyflow.observation import read_jsonl, utc_now_iso, write_jsonl

EXPOSURE_SIMULATION_SCHEMA_VERSION = "storyflow_echo_simulation_v1"
SUPPORTED_EXPOSURE_POLICIES = (
    "utility_only",
    "confidence_only",
    "utility_confidence",
    "cure_truce",
)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


@dataclass(frozen=True, slots=True)
class ExposureSimulationConfig:
    """Configuration for deterministic synthetic exposure simulation."""

    policies: tuple[str, ...] = SUPPORTED_EXPOSURE_POLICIES
    rounds: int = 3
    group_key: str = "input_id"
    exposures_per_group: int = 1
    confidence_source: str = DEFAULT_RERANK_CONFIDENCE_SOURCE
    utility_weight: float = 0.5
    confidence_weight: float = 0.5
    feedback_learning_rate: float = 0.2

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.exposures_per_group < 1:
            raise ValueError("exposures_per_group must be >= 1")
        unsupported = sorted(set(self.policies) - set(SUPPORTED_EXPOSURE_POLICIES))
        if unsupported:
            raise ValueError(f"unsupported exposure policies: {unsupported}")
        if self.confidence_source not in SUPPORTED_RERANK_CONFIDENCE_SOURCES:
            raise ValueError(f"unsupported confidence source: {self.confidence_source}")
        if self.utility_weight < 0 or self.confidence_weight < 0:
            raise ValueError("utility_weight and confidence_weight must be non-negative")
        if not 0 <= self.feedback_learning_rate <= 1:
            raise ValueError("feedback_learning_rate must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _SimulationCandidate:
    row_index: int
    row: Mapping[str, Any]
    group_id: str
    features: ExposureConfidenceFeatures
    initial_confidence: float
    confidence_source: str
    item_key: str
    category: str | None


def _group_id(row: Mapping[str, Any], features: ExposureConfidenceFeatures, group_key: str) -> str:
    value = row.get(group_key)
    if value in (None, ""):
        value = features.metadata.get(group_key)
    if value in (None, ""):
        raise ValueError(f"feature row is missing group key: {group_key}")
    return str(value)


def _category(row: Mapping[str, Any], features: ExposureConfidenceFeatures) -> str | None:
    metadata = row.get("metadata")
    feature_metadata = features.metadata
    for source in (metadata, feature_metadata, row):
        if not isinstance(source, Mapping):
            continue
        for key in (
            "generated_category",
            "category",
            "target_category",
            "main_category",
            "genres",
        ):
            value = source.get(key)
            if value not in (None, ""):
                return str(value)
    return None


def _item_key(candidate: _SimulationCandidate) -> str:
    if candidate.features.item_id:
        return str(candidate.features.item_id)
    return f"ungrounded::{candidate.group_id}::{candidate.row_index}"


def _load_candidates(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: ExposureSimulationConfig,
) -> list[_SimulationCandidate]:
    candidates: list[_SimulationCandidate] = []
    for row_index, row in enumerate(rows):
        features = feature_from_rerank_row(row)
        selected = select_rerank_confidence(
            row,
            confidence_source=config.confidence_source,
            strict_confidence_source=False,
        )
        group_id = _group_id(row, features, config.group_key)
        candidate = _SimulationCandidate(
            row_index=row_index,
            row=dict(row),
            group_id=group_id,
            features=features,
            initial_confidence=selected.value,
            confidence_source=selected.selected_source,
            item_key="",
            category=_category(row, features),
        )
        candidates.append(replace(candidate, item_key=_item_key(candidate)))
    if not candidates:
        raise ValueError("cannot simulate exposure over an empty feature row collection")
    return candidates


def _policy_score(
    *,
    policy: str,
    features: ExposureConfidenceFeatures,
    confidence: float,
    config: ExposureSimulationConfig,
    weights: CureTruceWeights,
) -> tuple[float, dict[str, float]]:
    confidence = _clip01(confidence)
    if policy == "utility_only":
        return features.preference_score, {"preference_score": features.preference_score}
    if policy == "confidence_only":
        return confidence, {"selected_confidence": confidence}
    if policy == "utility_confidence":
        total = config.utility_weight + config.confidence_weight
        if total <= 0:
            total = 1.0
        score = (
            config.utility_weight * features.preference_score
            + config.confidence_weight * confidence
        ) / total
        return score, {
            "preference_score": features.preference_score,
            "selected_confidence": confidence,
        }
    if policy == "cure_truce":
        risk_penalty = compute_risk_penalty(features, confidence, weights)
        echo_risk = compute_echo_risk(features, confidence)
        information_gain = compute_information_gain(features)
        score = (
            weights.preference_score_weight * features.preference_score
            + weights.exposure_confidence_weight * confidence
            + weights.information_gain_weight * information_gain
            - weights.risk_penalty_weight * risk_penalty
            - weights.echo_penalty_weight * echo_risk
        )
        return score, {
            "preference_score": features.preference_score,
            "selected_confidence": confidence,
            "risk_penalty": risk_penalty,
            "echo_risk": echo_risk,
            "information_gain": information_gain,
        }
    raise ValueError(f"unsupported exposure policy: {policy}")


def _feedback_proxy(features: ExposureConfidenceFeatures) -> tuple[float, str]:
    if features.correctness_label is not None:
        return float(features.correctness_label), "correctness_label_synthetic_proxy"
    return features.preference_score, "preference_score_synthetic_proxy"


def _gini(values: Iterable[int]) -> float:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return 0.0
    total = sum(ordered)
    if total <= 0:
        return 0.0
    n = len(ordered)
    weighted_sum = sum((index + 1) * value for index, value in enumerate(ordered))
    return max(0.0, (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n)


def _entropy(counts: Mapping[str, int]) -> float | None:
    total = sum(counts.values())
    if total <= 0:
        return None
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy


def _bucket(features: ExposureConfidenceFeatures) -> str:
    return str(features.popularity_bucket or "unknown")


def _round_summary(
    *,
    policy: str,
    round_index: int,
    candidate_item_keys: list[str],
    exposure_counts: Counter[str],
    bucket_counts: Counter[str],
    category_counts: Counter[str],
    selected_confidences: list[float],
    selected_feedback: list[float],
    state_confidences: Mapping[int, float],
    initial_state_mean: float,
) -> dict[str, Any]:
    total_exposures = sum(exposure_counts.values())
    tail_count = bucket_counts.get("tail", 0)
    head_count = bucket_counts.get("head", 0)
    mid_count = bucket_counts.get("mid", 0)
    return {
        "schema_version": EXPOSURE_SIMULATION_SCHEMA_VERSION,
        "policy": policy,
        "round": round_index,
        "total_exposures": total_exposures,
        "unique_exposed_items": sum(1 for count in exposure_counts.values() if count > 0),
        "candidate_item_count": len(candidate_item_keys),
        "exposure_gini": _gini(exposure_counts.get(key, 0) for key in candidate_item_keys),
        "head_exposure_share": head_count / total_exposures if total_exposures else 0.0,
        "mid_exposure_share": mid_count / total_exposures if total_exposures else 0.0,
        "tail_exposure_share": tail_count / total_exposures if total_exposures else 0.0,
        "unknown_bucket_exposure_share": bucket_counts.get("unknown", 0) / total_exposures
        if total_exposures
        else 0.0,
        "popularity_bucket_entropy": _entropy(bucket_counts),
        "category_entropy": _entropy(category_counts),
        "mean_exposed_confidence": sum(selected_confidences) / len(selected_confidences)
        if selected_confidences
        else None,
        "mean_synthetic_feedback_proxy": sum(selected_feedback) / len(selected_feedback)
        if selected_feedback
        else None,
        "mean_state_confidence": sum(state_confidences.values()) / len(state_confidences),
        "confidence_drift": (
            sum(state_confidences.values()) / len(state_confidences)
        )
        - initial_state_mean,
    }


def simulate_exposure_feedback_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: ExposureSimulationConfig | None = None,
    weights: CureTruceWeights | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Simulate confidence-guided exposure over serialized feature rows.

    The returned exposure records are synthetic diagnostics. They should be
    used to validate echo-risk plumbing and metric definitions, not to claim
    real user feedback or method performance.
    """

    config = config or ExposureSimulationConfig()
    weights = weights or CureTruceWeights()
    candidates = _load_candidates(rows, config=config)
    grouped: dict[str, list[_SimulationCandidate]] = {}
    group_order: list[str] = []
    for candidate in candidates:
        if candidate.group_id not in grouped:
            grouped[candidate.group_id] = []
            group_order.append(candidate.group_id)
        grouped[candidate.group_id].append(candidate)

    candidate_item_keys = sorted({candidate.item_key for candidate in candidates})
    exposure_records: list[dict[str, Any]] = []
    round_summaries: list[dict[str, Any]] = []
    final_policy_summaries: dict[str, dict[str, Any]] = {}

    for policy in config.policies:
        state_confidences = {
            candidate.row_index: candidate.initial_confidence for candidate in candidates
        }
        initial_state_mean = sum(state_confidences.values()) / len(state_confidences)
        exposure_counts: Counter[str] = Counter()
        bucket_counts: Counter[str] = Counter()
        category_counts: Counter[str] = Counter()

        for round_index in range(1, config.rounds + 1):
            selected_confidences: list[float] = []
            selected_feedback: list[float] = []
            for group_id in group_order:
                group_candidates = grouped[group_id]
                scored: list[tuple[float, str, str, int, _SimulationCandidate, dict[str, float]]] = []
                for candidate in group_candidates:
                    confidence = state_confidences[candidate.row_index]
                    score, components = _policy_score(
                        policy=policy,
                        features=candidate.features,
                        confidence=confidence,
                        config=config,
                        weights=weights,
                    )
                    scored.append(
                        (
                            score,
                            str(candidate.features.item_id or ""),
                            str(candidate.features.generated_title or ""),
                            candidate.row_index,
                            candidate,
                            components,
                        )
                    )
                scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
                for rank, scored_item in enumerate(scored[: config.exposures_per_group], start=1):
                    policy_score, _, _, _, candidate, components = scored_item
                    confidence_before = state_confidences[candidate.row_index]
                    feedback, feedback_source = _feedback_proxy(candidate.features)
                    confidence_after = _clip01(
                        confidence_before
                        + config.feedback_learning_rate * (feedback - confidence_before)
                    )
                    state_confidences[candidate.row_index] = confidence_after
                    exposure_counts[candidate.item_key] += 1
                    bucket = _bucket(candidate.features)
                    bucket_counts[bucket] += 1
                    if candidate.category is not None:
                        category_counts[candidate.category] += 1
                    selected_confidences.append(confidence_before)
                    selected_feedback.append(feedback)

                    output = dict(candidate.row)
                    output["is_experiment_result"] = False
                    output["exposure_simulation"] = {
                        "schema_version": EXPOSURE_SIMULATION_SCHEMA_VERSION,
                        "policy": policy,
                        "round": round_index,
                        "group_key": config.group_key,
                        "group_id": group_id,
                        "rank_within_group": rank,
                        "item_key": candidate.item_key,
                        "item_id": candidate.features.item_id,
                        "generated_title": candidate.features.generated_title,
                        "popularity_bucket": bucket,
                        "category": candidate.category,
                        "policy_score": policy_score,
                        "policy_components": components,
                        "selected_confidence_before_feedback": confidence_before,
                        "selected_confidence_after_feedback": confidence_after,
                        "selected_confidence_source": candidate.confidence_source,
                        "synthetic_feedback_proxy": feedback,
                        "synthetic_feedback_source": feedback_source,
                        "synthetic_feedback": True,
                        "feedback_learning_rate": config.feedback_learning_rate,
                        "echo_risk": compute_echo_risk(candidate.features, confidence_before),
                        "information_gain": compute_information_gain(candidate.features),
                        "risk_penalty": compute_risk_penalty(
                            candidate.features,
                            confidence_before,
                            weights,
                        ),
                        "api_called": False,
                        "model_training": False,
                        "server_executed": False,
                        "is_experiment_result": False,
                    }
                    exposure_records.append(output)

            summary = _round_summary(
                policy=policy,
                round_index=round_index,
                candidate_item_keys=candidate_item_keys,
                exposure_counts=exposure_counts,
                bucket_counts=bucket_counts,
                category_counts=category_counts,
                selected_confidences=selected_confidences,
                selected_feedback=selected_feedback,
                state_confidences=state_confidences,
                initial_state_mean=initial_state_mean,
            )
            round_summaries.append(summary)
            final_policy_summaries[policy] = summary

    summary = {
        "schema_version": EXPOSURE_SIMULATION_SCHEMA_VERSION,
        "config": config.to_dict(),
        "input_row_count": len(candidates),
        "group_count": len(grouped),
        "candidate_item_count": len(candidate_item_keys),
        "exposure_record_count": len(exposure_records),
        "round_summaries": round_summaries,
        "final_policy_summaries": final_policy_summaries,
        "synthetic_feedback": True,
        "api_called": False,
        "model_training": False,
        "server_executed": False,
        "is_experiment_result": False,
    }
    return exposure_records, summary


def simulate_exposure_feedback_jsonl(
    *,
    features_jsonl: str | Path,
    output_jsonl: str | Path,
    summary_json: str | Path,
    manifest_json: str | Path,
    config: ExposureSimulationConfig | None = None,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Run synthetic exposure simulation from feature JSONL and write artifacts."""

    features_path = Path(features_jsonl)
    rows = read_jsonl(features_path)
    if max_examples is not None:
        if max_examples < 1:
            raise ValueError("max_examples must be >= 1")
        rows = rows[:max_examples]
    exposure_rows, summary = simulate_exposure_feedback_rows(rows, config=config)
    write_jsonl(output_jsonl, exposure_rows)
    summary_path = Path(summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "created_at_utc": utc_now_iso(),
        **summary,
        "features_jsonl": str(features_path),
        "output_jsonl": str(output_jsonl),
        "summary_json": str(summary_path),
        "manifest_json": str(manifest_json),
        "max_examples": max_examples,
        "row_contract": {
            "input_requires": "CURE/TRUCE feature rows with grounded title provenance",
            "output_adds": "exposure_simulation",
            "synthetic_feedback_only": True,
            "generated_title_must_be_grounded_before_correctness": True,
        },
        "note": (
            "Synthetic confidence-guided exposure simulation. Feedback is a "
            "diagnostic proxy derived from existing labels or preference scores; "
            "this is not real user feedback, not model training, and not paper "
            "evidence."
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
    "EXPOSURE_SIMULATION_SCHEMA_VERSION",
    "SUPPORTED_EXPOSURE_POLICIES",
    "ExposureSimulationConfig",
    "simulate_exposure_feedback_jsonl",
    "simulate_exposure_feedback_rows",
]

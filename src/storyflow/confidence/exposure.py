"""Exposure-counterfactual confidence features and CURE/TRUCE scoring.

This module is the first Phase 4 scaffold. It defines typed feature and scoring
contracts around the target object

    C(u, i) ~= P(user accepts item i | user u, do(exposure=1)).

The functions below are deterministic heuristics for controlled tests and
pipeline integration. They are not a trained calibrator and must not be reported
as evidence that the framework improves recommendation quality.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from storyflow.schemas import PopularityBucket


def _clip01(value: float) -> float:
    """Clip finite numeric values to [0, 1], using 0.5 for non-finite values."""

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected numeric value, got {value!r}") from exc
    if not math.isfinite(numeric):
        return 0.5
    return min(1.0, max(0.0, numeric))


def _optional_clip01(value: float | None) -> float | None:
    if value is None:
        return None
    return _clip01(value)


def _copy_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return dict(metadata)


def _require_non_empty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _normalize_bucket(bucket: PopularityBucket | str | None) -> str | None:
    if bucket is None or bucket == "":
        return None
    if isinstance(bucket, PopularityBucket):
        return bucket.value
    normalized = str(bucket).strip().lower()
    if normalized not in {bucket.value for bucket in PopularityBucket}:
        raise ValueError("popularity_bucket must be one of head, mid, or tail")
    return normalized


def _bucket_popularity_prior(bucket: str | None) -> float:
    if bucket == PopularityBucket.HEAD.value:
        return 0.85
    if bucket == PopularityBucket.MID.value:
        return 0.50
    if bucket == PopularityBucket.TAIL.value:
        return 0.15
    return 0.50


@dataclass(frozen=True, slots=True)
class ExposureConfidenceFeatures:
    """Feature vector for estimating exposure-counterfactual confidence.

    The vector is meant to be built only after a generated title has passed
    through catalog grounding. If a generated title is ungrounded, keep
    ``item_id=None`` and ``is_grounded=False`` so the scorer can abstain.
    ``correctness_label`` is an observation/training label, not a default
    inference-time score input.
    """

    user_id: str
    item_id: str | None
    generated_title: str | None = None
    preference_score: float = 0.5
    verbal_confidence: float | None = None
    generation_confidence: float | None = None
    grounding_confidence: float | None = None
    grounding_ambiguity: float | None = None
    popularity_percentile: float | None = None
    popularity_bucket: PopularityBucket | str | None = None
    history_alignment: float | None = None
    novelty_score: float | None = None
    correctness_label: int | None = None
    is_grounded: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_string("user_id", self.user_id)
        if self.item_id is not None:
            _require_non_empty_string("item_id", self.item_id)
        if self.generated_title is not None and not self.generated_title.strip():
            raise ValueError("generated_title must be non-empty when provided")
        if self.correctness_label not in (None, 0, 1):
            raise ValueError("correctness_label must be None, 0, or 1")

        object.__setattr__(self, "preference_score", _clip01(self.preference_score))
        object.__setattr__(self, "verbal_confidence", _optional_clip01(self.verbal_confidence))
        object.__setattr__(
            self,
            "generation_confidence",
            _optional_clip01(self.generation_confidence),
        )
        object.__setattr__(
            self,
            "grounding_confidence",
            _optional_clip01(self.grounding_confidence),
        )
        object.__setattr__(
            self,
            "grounding_ambiguity",
            _optional_clip01(self.grounding_ambiguity),
        )
        object.__setattr__(
            self,
            "popularity_percentile",
            _optional_clip01(self.popularity_percentile),
        )
        object.__setattr__(
            self,
            "popularity_bucket",
            _normalize_bucket(self.popularity_bucket),
        )
        object.__setattr__(
            self,
            "history_alignment",
            _optional_clip01(self.history_alignment),
        )
        object.__setattr__(self, "novelty_score", _optional_clip01(self.novelty_score))
        if self.item_id is None:
            object.__setattr__(self, "is_grounded", False)
        object.__setattr__(self, "metadata", _copy_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class CureTruceWeights:
    """Deterministic scaffold weights for CURE/TRUCE scoring.

    These weights make the code path testable. They are not learned parameters
    and should be replaced or calibrated from observation data before any
    method-level claim.
    """

    preference_evidence_weight: float = 0.25
    verbal_evidence_weight: float = 0.25
    generation_evidence_weight: float = 0.15
    grounding_evidence_weight: float = 0.25
    history_evidence_weight: float = 0.10
    grounding_ambiguity_evidence_penalty: float = 0.20
    ungrounded_confidence_multiplier: float = 0.20

    preference_score_weight: float = 0.45
    exposure_confidence_weight: float = 0.35
    information_gain_weight: float = 0.20
    risk_penalty_weight: float = 0.55
    echo_penalty_weight: float = 0.45

    ungrounded_risk_penalty: float = 1.00
    ambiguity_risk_weight: float = 0.35
    overclaim_risk_weight: float = 0.35
    wrong_high_confidence_risk_weight: float = 0.20

    abstain_risk_threshold: float = 0.75
    diversify_echo_threshold: float = 0.35
    recommend_confidence_threshold: float = 0.50
    explore_information_threshold: float = 0.20


@dataclass(frozen=True, slots=True)
class CureTruceScore:
    """Scored candidate produced by the CURE/TRUCE scaffold."""

    item_id: str | None
    score: float
    estimated_exposure_confidence: float
    risk_penalty: float
    echo_risk: float
    information_gain: float
    popularity_residual: float
    action: str
    components: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "estimated_exposure_confidence",
            _clip01(self.estimated_exposure_confidence),
        )
        object.__setattr__(self, "risk_penalty", _clip01(self.risk_penalty))
        object.__setattr__(self, "echo_risk", _clip01(self.echo_risk))
        object.__setattr__(self, "information_gain", _clip01(self.information_gain))
        object.__setattr__(self, "components", _copy_metadata(self.components))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


def popularity_pressure(features: ExposureConfidenceFeatures) -> float:
    """Return a [0, 1] popularity pressure value for echo-risk accounting."""

    if features.popularity_percentile is not None:
        return features.popularity_percentile
    return _bucket_popularity_prior(features.popularity_bucket)


def estimate_exposure_confidence(
    features: ExposureConfidenceFeatures,
    weights: CureTruceWeights | None = None,
) -> float:
    """Estimate the scaffold proxy for C(u, i).

    The estimate dynamically averages available evidence: base preference,
    verbal confidence, generation evidence, grounding confidence, and history
    alignment. Grounding ambiguity and ungrounded status reduce the estimate
    because a title that cannot be mapped to a catalog item cannot support a
    reliable catalog-level recommendation.
    """

    weights = weights or CureTruceWeights()
    evidence: list[tuple[float, float]] = [
        (features.preference_score, weights.preference_evidence_weight)
    ]
    if features.verbal_confidence is not None:
        evidence.append((features.verbal_confidence, weights.verbal_evidence_weight))
    if features.generation_confidence is not None:
        evidence.append((features.generation_confidence, weights.generation_evidence_weight))
    if features.grounding_confidence is not None:
        evidence.append((features.grounding_confidence, weights.grounding_evidence_weight))
    if features.history_alignment is not None:
        evidence.append((features.history_alignment, weights.history_evidence_weight))

    total_weight = sum(weight for _, weight in evidence if weight > 0)
    if total_weight <= 0:
        estimate = 0.5
    else:
        estimate = sum(value * weight for value, weight in evidence if weight > 0) / total_weight

    ambiguity = features.grounding_ambiguity or 0.0
    estimate -= weights.grounding_ambiguity_evidence_penalty * ambiguity
    if not features.is_grounded:
        estimate *= weights.ungrounded_confidence_multiplier
    return _clip01(estimate)


def compute_popularity_residual_confidence(
    features: ExposureConfidenceFeatures,
    weights: CureTruceWeights | None = None,
) -> float:
    """Return confidence unexplained by the item's popularity prior.

    Positive values mean the current confidence proxy is higher than the
    popularity prior; negative values mean the item is less trusted than a
    simple popularity-only expectation.
    """

    confidence = (
        features.verbal_confidence
        if features.verbal_confidence is not None
        else estimate_exposure_confidence(features, weights)
    )
    return confidence - popularity_pressure(features)


def compute_echo_risk(
    features: ExposureConfidenceFeatures,
    estimated_confidence: float | None = None,
) -> float:
    """Estimate echo risk from confidence, popularity, and low novelty."""

    confidence = (
        estimate_exposure_confidence(features)
        if estimated_confidence is None
        else _clip01(estimated_confidence)
    )
    popularity = popularity_pressure(features)
    novelty = features.novelty_score
    if novelty is None and features.history_alignment is not None:
        novelty = 1.0 - features.history_alignment
    if novelty is None:
        novelty = 0.5
    familiarity = features.history_alignment if features.history_alignment is not None else 1.0 - novelty
    grounding_factor = 1.0 if features.is_grounded else 0.5
    low_novelty_pressure = 0.5 * (1.0 - novelty) + 0.5 * familiarity
    return _clip01(confidence * popularity * low_novelty_pressure * grounding_factor)


def compute_information_gain(features: ExposureConfidenceFeatures) -> float:
    """Return a simple exploration value for grounded novel tail candidates."""

    novelty = features.novelty_score
    if novelty is None and features.history_alignment is not None:
        novelty = 1.0 - features.history_alignment
    if novelty is None:
        novelty = 0.5
    grounding = features.grounding_confidence
    if grounding is None:
        grounding = 1.0 if features.is_grounded else 0.0
    return _clip01(novelty * (1.0 - popularity_pressure(features)) * grounding)


def compute_risk_penalty(
    features: ExposureConfidenceFeatures,
    estimated_confidence: float,
    weights: CureTruceWeights | None = None,
) -> float:
    """Compute abstention/penalty risk for unsafe confidence use."""

    weights = weights or CureTruceWeights()
    if not features.is_grounded:
        return _clip01(weights.ungrounded_risk_penalty)

    ambiguity = features.grounding_ambiguity or 0.0
    grounding = features.grounding_confidence
    if grounding is None:
        grounding = 1.0
    verbal = features.verbal_confidence
    if verbal is None:
        verbal = estimated_confidence
    overclaim = max(0.0, verbal - grounding)

    wrong_high_confidence = 0.0
    if features.correctness_label == 0 and verbal >= 0.70:
        wrong_high_confidence = verbal

    return _clip01(
        weights.ambiguity_risk_weight * ambiguity
        + weights.overclaim_risk_weight * overclaim
        + weights.wrong_high_confidence_risk_weight * wrong_high_confidence
    )


def _choose_action(
    *,
    features: ExposureConfidenceFeatures,
    estimated_confidence: float,
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
    if estimated_confidence >= weights.recommend_confidence_threshold:
        return "recommend"
    if information_gain >= weights.explore_information_threshold:
        return "explore"
    return "abstain"


def score_cure_truce_candidate(
    features: ExposureConfidenceFeatures,
    weights: CureTruceWeights | None = None,
) -> CureTruceScore:
    """Score one grounded candidate with the CURE/TRUCE scaffold."""

    weights = weights or CureTruceWeights()
    estimated = estimate_exposure_confidence(features, weights)
    risk_penalty = compute_risk_penalty(features, estimated, weights)
    echo_risk = compute_echo_risk(features, estimated)
    information_gain = compute_information_gain(features)
    residual = compute_popularity_residual_confidence(features, weights)
    score = (
        weights.preference_score_weight * features.preference_score
        + weights.exposure_confidence_weight * estimated
        + weights.information_gain_weight * information_gain
        - weights.risk_penalty_weight * risk_penalty
        - weights.echo_penalty_weight * echo_risk
    )
    action = _choose_action(
        features=features,
        estimated_confidence=estimated,
        risk_penalty=risk_penalty,
        echo_risk=echo_risk,
        information_gain=information_gain,
        weights=weights,
    )
    return CureTruceScore(
        item_id=features.item_id,
        score=score,
        estimated_exposure_confidence=estimated,
        risk_penalty=risk_penalty,
        echo_risk=echo_risk,
        information_gain=information_gain,
        popularity_residual=residual,
        action=action,
        components={
            "preference_score": features.preference_score,
            "verbal_confidence": features.verbal_confidence,
            "generation_confidence": features.generation_confidence,
            "grounding_confidence": features.grounding_confidence,
            "grounding_ambiguity": features.grounding_ambiguity,
            "popularity_pressure": popularity_pressure(features),
            "popularity_bucket": features.popularity_bucket,
            "history_alignment": features.history_alignment,
            "novelty_score": features.novelty_score,
            "correctness_label": features.correctness_label,
            "is_grounded": features.is_grounded,
        },
    )


def rerank_cure_truce(
    candidates: Iterable[ExposureConfidenceFeatures],
    weights: CureTruceWeights | None = None,
    *,
    top_k: int | None = None,
) -> list[tuple[ExposureConfidenceFeatures, CureTruceScore]]:
    """Score and sort candidates by CURE/TRUCE score.

    Sorting is deterministic: score descending, then item id/title, then input
    order. The function does not mutate candidates and does not train or call
    external services.
    """

    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1 when provided")
    weights = weights or CureTruceWeights()
    scored = [
        (index, candidate, score_cure_truce_candidate(candidate, weights))
        for index, candidate in enumerate(candidates)
    ]
    scored.sort(
        key=lambda item: (
            -item[2].score,
            str(item[1].item_id or ""),
            str(item[1].generated_title or ""),
            item[0],
        )
    )
    pairs = [(candidate, score) for _, candidate, score in scored]
    if top_k is not None:
        return pairs[:top_k]
    return pairs


__all__ = [
    "CureTruceScore",
    "CureTruceWeights",
    "ExposureConfidenceFeatures",
    "compute_echo_risk",
    "compute_information_gain",
    "compute_popularity_residual_confidence",
    "compute_risk_penalty",
    "estimate_exposure_confidence",
    "popularity_pressure",
    "rerank_cure_truce",
    "score_cure_truce_candidate",
]

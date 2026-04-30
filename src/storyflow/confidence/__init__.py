"""Confidence extraction, calibration, and exposure-aware scoring modules."""

from storyflow.confidence.exposure import (
    CureTruceScore,
    CureTruceWeights,
    ExposureConfidenceFeatures,
    compute_echo_risk,
    compute_information_gain,
    compute_popularity_residual_confidence,
    compute_risk_penalty,
    estimate_exposure_confidence,
    popularity_pressure,
    rerank_cure_truce,
    score_cure_truce_candidate,
)
from storyflow.confidence.features import (
    FEATURE_SCHEMA_VERSION,
    CatalogFeatureIndex,
    build_confidence_features,
    confidence_feature_record,
    feature_from_grounded_row,
)

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "CatalogFeatureIndex",
    "CureTruceScore",
    "CureTruceWeights",
    "ExposureConfidenceFeatures",
    "build_confidence_features",
    "confidence_feature_record",
    "compute_echo_risk",
    "compute_information_gain",
    "compute_popularity_residual_confidence",
    "compute_risk_penalty",
    "estimate_exposure_confidence",
    "feature_from_grounded_row",
    "popularity_pressure",
    "rerank_cure_truce",
    "score_cure_truce_candidate",
]

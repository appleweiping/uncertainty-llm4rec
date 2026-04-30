from __future__ import annotations

import pytest

from storyflow.confidence import (
    CureTruceWeights,
    ExposureConfidenceFeatures,
    compute_echo_risk,
    compute_popularity_residual_confidence,
    estimate_exposure_confidence,
    rerank_cure_truce,
    score_cure_truce_candidate,
)
from storyflow.schemas import PopularityBucket


def test_exposure_confidence_features_clip_values_and_normalize_bucket() -> None:
    features = ExposureConfidenceFeatures(
        user_id="u1",
        item_id=None,
        preference_score=2.0,
        verbal_confidence=-0.2,
        grounding_confidence=float("inf"),
        popularity_percentile=1.3,
        popularity_bucket=PopularityBucket.TAIL,
    )

    assert features.preference_score == 1.0
    assert features.verbal_confidence == 0.0
    assert features.grounding_confidence == 0.5
    assert features.popularity_percentile == 1.0
    assert features.popularity_bucket == "tail"
    assert features.is_grounded is False


def test_exposure_confidence_features_reject_bad_labels_and_buckets() -> None:
    with pytest.raises(ValueError, match="correctness_label"):
        ExposureConfidenceFeatures(user_id="u1", item_id="i1", correctness_label=2)

    with pytest.raises(ValueError, match="popularity_bucket"):
        ExposureConfidenceFeatures(user_id="u1", item_id="i1", popularity_bucket="viral")


def test_ungrounded_candidate_is_low_confidence_and_abstains() -> None:
    features = ExposureConfidenceFeatures(
        user_id="u1",
        item_id=None,
        generated_title="Unmatched Product",
        preference_score=0.9,
        verbal_confidence=0.95,
        generation_confidence=0.9,
        grounding_confidence=0.0,
        grounding_ambiguity=1.0,
        popularity_bucket="tail",
        is_grounded=False,
    )

    estimated = estimate_exposure_confidence(features)
    scored = score_cure_truce_candidate(features)

    assert estimated < 0.2
    assert scored.action == "abstain"
    assert scored.risk_penalty == 1.0
    assert scored.score < 0.0


def test_popular_familiar_high_confidence_candidate_has_higher_echo_risk() -> None:
    head = ExposureConfidenceFeatures(
        user_id="u1",
        item_id="head",
        preference_score=0.8,
        verbal_confidence=0.92,
        grounding_confidence=0.95,
        popularity_percentile=0.95,
        history_alignment=0.9,
        novelty_score=0.05,
        popularity_bucket="head",
    )
    tail = ExposureConfidenceFeatures(
        user_id="u1",
        item_id="tail",
        preference_score=0.8,
        verbal_confidence=0.86,
        grounding_confidence=0.95,
        popularity_percentile=0.10,
        history_alignment=0.2,
        novelty_score=0.85,
        popularity_bucket="tail",
    )

    assert compute_echo_risk(head) > compute_echo_risk(tail)
    assert score_cure_truce_candidate(head).action == "diversify"


def test_popularity_residual_tracks_confidence_above_or_below_popularity_prior() -> None:
    confident_tail = ExposureConfidenceFeatures(
        user_id="u1",
        item_id="tail",
        verbal_confidence=0.75,
        popularity_percentile=0.10,
        popularity_bucket="tail",
    )
    underconfident_head = ExposureConfidenceFeatures(
        user_id="u1",
        item_id="head",
        verbal_confidence=0.35,
        popularity_percentile=0.90,
        popularity_bucket="head",
    )

    assert compute_popularity_residual_confidence(confident_tail) > 0.0
    assert compute_popularity_residual_confidence(underconfident_head) < 0.0


def test_rerank_prefers_safer_novel_tail_when_echo_penalty_is_high() -> None:
    head = ExposureConfidenceFeatures(
        user_id="u1",
        item_id="head",
        generated_title="Head Serum",
        preference_score=0.82,
        verbal_confidence=0.94,
        grounding_confidence=0.95,
        grounding_ambiguity=0.02,
        popularity_percentile=0.96,
        popularity_bucket="head",
        history_alignment=0.9,
        novelty_score=0.05,
    )
    tail = ExposureConfidenceFeatures(
        user_id="u1",
        item_id="tail",
        generated_title="Tail Balm",
        preference_score=0.74,
        verbal_confidence=0.72,
        generation_confidence=0.70,
        grounding_confidence=0.90,
        grounding_ambiguity=0.05,
        popularity_percentile=0.08,
        popularity_bucket="tail",
        history_alignment=0.25,
        novelty_score=0.82,
    )
    weights = CureTruceWeights(echo_penalty_weight=0.70, information_gain_weight=0.25)

    ranked = rerank_cure_truce([head, tail], weights)

    assert ranked[0][0].item_id == "tail"
    assert ranked[0][1].information_gain > ranked[1][1].information_gain
    assert ranked[1][1].echo_risk > ranked[0][1].echo_risk


def test_rerank_is_deterministic_and_respects_top_k() -> None:
    left = ExposureConfidenceFeatures(user_id="u1", item_id="a", preference_score=0.5)
    right = ExposureConfidenceFeatures(user_id="u1", item_id="b", preference_score=0.5)

    ranked = rerank_cure_truce([right, left], top_k=1)

    assert len(ranked) == 1
    assert ranked[0][0].item_id == "a"

    with pytest.raises(ValueError, match="top_k"):
        rerank_cure_truce([left], top_k=0)

from __future__ import annotations

from llm4rec.methods.uncertainty_policy import UncertaintyPolicy


def _signals(**overrides):
    values = {
        "parse_success": True,
        "confidence": 0.82,
        "grounding_success": True,
        "grounding_score": 1.0,
        "candidate_normalized_confidence": 0.7,
        "popularity_bucket": "tail",
        "history_similarity": 0.2,
    }
    values.update(overrides)
    return values


def test_policy_accepts_when_confidence_and_grounding_pass() -> None:
    decision = UncertaintyPolicy().decide(_signals())
    assert decision.decision == "accept"
    assert "confidence_and_grounding_pass" in decision.reasons


def test_policy_falls_back_on_low_confidence() -> None:
    decision = UncertaintyPolicy().decide(_signals(confidence=0.2))
    assert decision.decision == "fallback"
    assert decision.reasons == ["low_confidence"]


def test_policy_reranks_high_confidence_head_item() -> None:
    decision = UncertaintyPolicy().decide(_signals(confidence=0.95, popularity_bucket="head"))
    assert decision.decision == "rerank"
    assert decision.risk_flags["head_item_overconfidence"] is True


def test_policy_reranks_echo_risk_when_guard_enabled() -> None:
    decision = UncertaintyPolicy().decide(_signals(confidence=0.95, history_similarity=0.9))
    assert decision.decision == "rerank"
    assert decision.risk_flags["echo_risk"] is True


def test_policy_abstains_when_fallback_disabled() -> None:
    policy = UncertaintyPolicy(components={"fallback": False})
    decision = policy.decide(_signals(confidence=0.1))
    assert decision.decision == "abstain"
    assert "fallback_disabled" in decision.reasons


def test_policy_without_uncertainty_accepts_grounded_parse() -> None:
    policy = UncertaintyPolicy(components={"uncertainty_policy": False})
    decision = policy.decide(_signals(confidence=0.1))
    assert decision.decision == "accept"
    assert decision.reasons == ["uncertainty_policy_disabled"]


def test_require_candidate_adherent_fallback() -> None:
    policy = UncertaintyPolicy(policy={"require_candidate_adherent": True})
    decision = policy.decide(
        _signals(
            grounded_item_in_candidates=False,
            confidence=0.96,
            candidate_normalized_confidence=0.96,
        )
    )
    assert decision.decision == "fallback"
    assert "not_candidate_adherent" in decision.reasons


def test_rerank_disabled_maps_to_fallback() -> None:
    policy = UncertaintyPolicy(policy={"enable_rerank_override": False})
    decision = policy.decide(_signals(confidence=0.95, popularity_bucket="head"))
    assert decision.decision == "fallback"
    assert "rerank_disabled_treat_as_fallback" in decision.reasons

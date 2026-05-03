from __future__ import annotations

from llm4rec.analysis.policy_sweep import PolicyVariant, replay_policy_row, should_override


def _base_row(decision: str, confidence: float, grounded: str = "a") -> dict:
    return {
        "user_id": "u",
        "target_item": "t",
        "candidate_items": ["a", "b", "t"],
        "predicted_items": [grounded, "b", "t"],
        "scores": [1.0, 0.5, 0.1],
        "method": "ours_uncertainty_guided_real",
        "domain": "movies",
        "raw_output": "{}",
        "metadata": {
            "example_id": "u:1",
            "uncertainty_decision": decision,
            "confidence": confidence,
            "grounding_score": 1.0,
            "candidate_adherent": True,
            "grounded_item_id": grounded,
            "candidate_normalized_confidence": 0.8,
        },
    }


def test_conservative_policy_rejects_low_confidence_override() -> None:
    ours = _base_row("accept", 0.85)
    variant = PolicyVariant(
        name="ours_conservative_uncertainty_gate",
        min_accept_confidence=0.95,
        min_grounding_score=1.0,
        require_candidate_adherent=True,
        require_candidate_normalized=True,
        enable_rerank=False,
    )
    assert should_override(ours, variant) is False


def test_replay_policy_preserves_fallback_when_rejected() -> None:
    ours = _base_row("accept", 0.85)
    fallback = {**ours, "method": "ours_fallback_only", "predicted_items": ["t", "a", "b"], "metadata": {"example_id": "u:1"}}
    variant = PolicyVariant(name="strict", min_accept_confidence=0.95)
    replayed = replay_policy_row(ours, fallback, variant)
    assert replayed["predicted_items"] == ["t", "a", "b"]
    assert replayed["metadata"]["offline_policy_source"] == "fallback"


def test_candidate_adherent_confident_accept_can_override() -> None:
    ours = _base_row("accept", 0.99)
    variant = PolicyVariant(name="strict", min_accept_confidence=0.95, require_candidate_adherent=True)
    assert should_override(ours, variant) is True


from __future__ import annotations

from typing import Any

from src.shadow.scoring import compute_shadow_scores


def build_shadow_signal_target(
    sample: dict[str, Any],
    *,
    variant: str,
    teacher_score: float | None = None,
    mu: float = 0.6,
) -> dict[str, Any]:
    """Build a leakage-safe first-pass target for shadow Signal LoRA.

    The target intentionally uses only the sample label plus optional training
    teacher score/proxies supplied by the caller. Test labels should never be
    passed here when constructing train data.
    """

    label = float(sample.get("label", 0.0) or 0.0)
    if teacher_score is None:
        target_probability = label
    else:
        target_probability = mu * label + (1.0 - mu) * float(teacher_score)
    target_probability = max(0.0, min(1.0, target_probability))

    popularity_group = str(
        sample.get("target_popularity_group")
        or sample.get("candidate_popularity_group")
        or "unknown"
    ).lower()
    title = str(sample.get("candidate_title") or sample.get("title") or "").strip()
    candidate_meta = str(
        sample.get("candidate_meta")
        or sample.get("candidate_description")
        or sample.get("candidate_text")
        or ""
    ).strip()

    evidence_support = 0.65 if label >= 0.5 else 0.25
    if title or candidate_meta:
        evidence_support += 0.1
    if popularity_group == "tail" and label >= 0.5:
        evidence_support += 0.05

    counterevidence_strength = 0.2 if label >= 0.5 else 0.65
    if popularity_group == "head" and label < 0.5:
        counterevidence_strength += 0.05

    target = {
        "relevance_probability": target_probability,
        "topk_inclusion_probability": target_probability,
        "preference_strength": target_probability,
        "expected_rank_percentile": 1.0 - target_probability,
        "match_probability": target_probability,
        "evidence_support": max(0.0, min(1.0, evidence_support)),
        "counterevidence_strength": max(0.0, min(1.0, counterevidence_strength)),
        "facet_alignment": max(0.0, min(1.0, evidence_support)),
        "facet_conflict": max(0.0, min(1.0, counterevidence_strength)),
        "history_support": max(0.0, min(1.0, evidence_support)),
        "novelty_pressure": 0.45 if popularity_group == "tail" else 0.25,
        "cutoff_margin_estimate": 2.0 * target_probability - 1.0,
        "competitive_pressure": 0.5,
        "rank_entropy": 1.0 - abs(2.0 * target_probability - 1.0),
        "frontier_probability": target_probability,
        "rank_confidence": abs(2.0 * target_probability - 1.0),
        "intent_prototype": "short user-interest prototype",
        "prototype_confidence": max(0.0, min(1.0, evidence_support)),
        "match_evidence": max(0.0, min(1.0, evidence_support)),
        "mismatch_strength": max(0.0, min(1.0, counterevidence_strength)),
        "reason": "The target is built from training labels and non-test support proxies.",
    }
    target.update(compute_shadow_scores(target, variant=variant))
    return target

from __future__ import annotations

from math import exp
from typing import Any

from src.shadow.schema import get_shadow_variant


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe01(record: dict[str, Any], field: str, default: float = 0.0) -> float:
    try:
        value = float(record.get(field, default))
    except Exception:
        value = default
    if value < 0:
        value = default
    return _clamp01(value)


def _risk_adjusted(score: float, uncertainty: float, eta: float) -> float:
    return _clamp01(score) * ((1.0 - _clamp01(uncertainty)) ** max(0.0, float(eta)))


def compute_shadow_scores(
    record: dict[str, Any],
    *,
    variant: str,
    calibrated_score: float | None = None,
    eta: float = 1.0,
) -> dict[str, float]:
    """Compute first-pass shadow uncertainty and risk-adjusted utility.

    This intentionally stays lightweight: calibration, budget consistency, and
    listwise margins are added upstream when validation outputs exist. These
    formulas give every shadow variant a stable common scoring contract.
    """

    spec = get_shadow_variant(variant)
    raw_score = _safe01(record, spec.primary_score_field, default=0.0)
    score = _clamp01(calibrated_score if calibrated_score is not None else raw_score)

    if spec.variant == "shadow_v1":
        evidence = _safe01(record, "evidence_support")
        counter = _safe01(record, "counterevidence_strength")
        boundary = 4.0 * score * (1.0 - score)
        cal_gap = abs(raw_score - score)
        evidence_risk = 1.0 - _clamp01(evidence - counter)
        uncertainty = _clamp01(0.5 * boundary + 0.3 * cal_gap + 0.2 * evidence_risk)
    elif spec.variant == "shadow_v2":
        margin = abs(float(record.get("cutoff_margin_estimate", 0.0) or 0.0))
        competitive_pressure = _safe01(record, "competitive_pressure")
        cal_gap = abs(raw_score - score)
        cutoff_risk = exp(-margin)
        competitive_risk = competitive_pressure * (1.0 - abs(2.0 * score - 1.0))
        uncertainty = _clamp01(0.45 * cutoff_risk + 0.3 * cal_gap + 0.25 * competitive_risk)
    elif spec.variant == "shadow_v3":
        facet_conflict = _safe01(record, "facet_conflict")
        history_support = _safe01(record, "history_support")
        novelty_pressure = _safe01(record, "novelty_pressure")
        uncertainty = _clamp01(0.4 * facet_conflict + 0.35 * (1.0 - history_support) + 0.25 * novelty_pressure)
    elif spec.variant == "shadow_v4":
        rank_entropy = _safe01(record, "rank_entropy")
        rank_confidence = _safe01(record, "rank_confidence")
        frontier_probability = _safe01(record, "frontier_probability")
        frontier_risk = 4.0 * frontier_probability * (1.0 - frontier_probability)
        utility_score = (1.0 - score) * frontier_probability
        uncertainty = _clamp01(0.45 * rank_entropy + 0.35 * frontier_risk + 0.2 * (1.0 - rank_confidence))
        return {
            "shadow_raw_score": raw_score,
            "shadow_score": _clamp01(utility_score),
            "shadow_uncertainty": uncertainty,
            "shadow_risk_adjusted_score": _risk_adjusted(utility_score, uncertainty, eta),
        }
    elif spec.variant == "shadow_v5":
        prototype_confidence = _safe01(record, "prototype_confidence")
        match_evidence = _safe01(record, "match_evidence")
        mismatch_strength = _safe01(record, "mismatch_strength")
        utility_score = score * prototype_confidence
        uncertainty = _clamp01(0.35 * (1.0 - prototype_confidence) + 0.35 * (1.0 - match_evidence) + 0.3 * mismatch_strength)
        return {
            "shadow_raw_score": raw_score,
            "shadow_score": _clamp01(utility_score),
            "shadow_uncertainty": uncertainty,
            "shadow_risk_adjusted_score": _risk_adjusted(utility_score, uncertainty, eta),
        }
    elif spec.variant == "shadow_v6":
        signal_score = _safe01(record, "signal_score")
        signal_uncertainty = _safe01(record, "signal_uncertainty")
        correction_gate = _safe01(record, "correction_gate")
        anchor_score = _safe01(record, "anchor_score", default=score)
        utility_score = correction_gate * signal_score + (1.0 - correction_gate) * anchor_score
        uncertainty = _clamp01(0.65 * signal_uncertainty + 0.35 * (1.0 - correction_gate))
        return {
            "shadow_raw_score": raw_score,
            "shadow_score": _clamp01(utility_score),
            "shadow_uncertainty": uncertainty,
            "shadow_risk_adjusted_score": _risk_adjusted(utility_score, uncertainty, eta),
        }
    else:
        uncertainty = 1.0 - score

    return {
        "shadow_raw_score": raw_score,
        "shadow_score": score,
        "shadow_uncertainty": uncertainty,
        "shadow_risk_adjusted_score": _risk_adjusted(score, uncertainty, eta),
    }

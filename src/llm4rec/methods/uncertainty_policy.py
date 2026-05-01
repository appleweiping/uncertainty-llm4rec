"""Uncertainty policy for Calibrated Uncertainty-Guided Recommendation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_DECISIONS = {"accept", "fallback", "abstain", "rerank"}


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    decision: str
    reasons: list[str] = field(default_factory=list)
    risk_flags: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        if self.decision not in VALID_DECISIONS:
            raise ValueError(f"invalid policy decision: {self.decision}")
        return {
            "decision": self.decision,
            "reasons": list(self.reasons),
            "risk_flags": dict(self.risk_flags),
        }


class UncertaintyPolicy:
    """Config-driven routing policy that never inspects ground-truth labels."""

    def __init__(
        self,
        *,
        thresholds: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
        components: dict[str, bool] | None = None,
    ) -> None:
        self.thresholds = {
            "min_accept_confidence": 0.7,
            "min_grounding_score": 0.8,
            "min_candidate_normalized_confidence": 0.5,
            "high_confidence": 0.85,
            "echo_similarity_threshold": 0.8,
            **(thresholds or {}),
        }
        self.policy = {
            "allow_abstain": True,
            "fallback_on_low_confidence": True,
            "fallback_on_grounding_failure": True,
            "penalize_head_item_overconfidence": True,
            "echo_risk_guard": True,
            **(policy or {}),
        }
        self.components = {
            "uncertainty_policy": True,
            "grounding_check": True,
            "candidate_normalized_confidence": True,
            "popularity_adjustment": True,
            "echo_risk_guard": True,
            "fallback": True,
            "fallback_only": False,
            **(components or {}),
        }

    def decide(self, signals: dict[str, Any]) -> PolicyDecision:
        if self.components.get("fallback_only", False):
            return self._fallback_decision(["fallback_only_ablation"])

        reasons: list[str] = []
        risk_flags = self._risk_flags(signals)

        if not bool(signals.get("parse_success", False)):
            return self._fallback_decision(["parse_failure"], risk_flags=risk_flags)

        if not self.components.get("uncertainty_policy", True):
            return self._decision_without_uncertainty(signals, risk_flags=risk_flags)

        if self.components.get("grounding_check", True):
            if not bool(signals.get("grounding_success", False)):
                return self._fallback_decision(["grounding_failure"], risk_flags=risk_flags)
            if float(signals.get("grounding_score") or 0.0) < self._threshold("min_grounding_score"):
                return self._fallback_decision(["low_grounding_score"], risk_flags=risk_flags)

        confidence = float(signals.get("confidence") or 0.0)
        if confidence < self._threshold("min_accept_confidence"):
            if bool(self.policy.get("fallback_on_low_confidence", True)):
                return self._fallback_decision(["low_confidence"], risk_flags=risk_flags)
            reasons.append("low_confidence")

        if self.components.get("candidate_normalized_confidence", True):
            normalized = signals.get("candidate_normalized_confidence")
            if normalized is None:
                return self._fallback_decision(["candidate_normalized_confidence_missing"], risk_flags=risk_flags)
            if float(normalized) < self._threshold("min_candidate_normalized_confidence"):
                return self._fallback_decision(["low_candidate_normalized_confidence"], risk_flags=risk_flags)

        if risk_flags.get("head_item_overconfidence", False):
            return self._rerank_or_fallback(["head_item_overconfidence"], risk_flags=risk_flags)
        if risk_flags.get("echo_risk", False):
            return self._rerank_or_fallback(["echo_risk"], risk_flags=risk_flags)

        return PolicyDecision("accept", reasons or ["confidence_and_grounding_pass"], risk_flags)

    def _decision_without_uncertainty(
        self,
        signals: dict[str, Any],
        *,
        risk_flags: dict[str, bool],
    ) -> PolicyDecision:
        if self.components.get("grounding_check", True) and not bool(signals.get("grounding_success", False)):
            return self._fallback_decision(["grounding_failure_without_uncertainty"], risk_flags=risk_flags)
        return PolicyDecision("accept", ["uncertainty_policy_disabled"], risk_flags)

    def _risk_flags(self, signals: dict[str, Any]) -> dict[str, bool]:
        confidence = float(signals.get("confidence") or 0.0)
        high_confidence = confidence >= self._threshold("high_confidence")
        popularity_bucket = str(signals.get("popularity_bucket") or "unknown")
        history_similarity = float(signals.get("history_similarity") or 0.0)
        head_overconfidence = (
            high_confidence
            and popularity_bucket == "head"
            and self.components.get("popularity_adjustment", True)
            and bool(self.policy.get("penalize_head_item_overconfidence", True))
        )
        echo_risk = (
            high_confidence
            and history_similarity >= self._threshold("echo_similarity_threshold")
            and self.components.get("echo_risk_guard", True)
            and bool(self.policy.get("echo_risk_guard", True))
        )
        return {
            "high_confidence": high_confidence,
            "head_item_overconfidence": head_overconfidence,
            "echo_risk": echo_risk,
        }

    def _fallback_decision(
        self,
        reasons: list[str],
        *,
        risk_flags: dict[str, bool] | None = None,
    ) -> PolicyDecision:
        if self.components.get("fallback", True):
            return PolicyDecision("fallback", reasons, risk_flags or {})
        if bool(self.policy.get("allow_abstain", True)):
            return PolicyDecision("abstain", [*reasons, "fallback_disabled"], risk_flags or {})
        return PolicyDecision("accept", [*reasons, "fallback_and_abstain_disabled"], risk_flags or {})

    def _rerank_or_fallback(
        self,
        reasons: list[str],
        *,
        risk_flags: dict[str, bool],
    ) -> PolicyDecision:
        if self.components.get("fallback", True):
            return PolicyDecision("rerank", reasons, risk_flags)
        if bool(self.policy.get("allow_abstain", True)):
            return PolicyDecision("abstain", [*reasons, "fallback_disabled"], risk_flags)
        return PolicyDecision("accept", [*reasons, "fallback_and_abstain_disabled"], risk_flags)

    def _threshold(self, name: str) -> float:
        return float(self.thresholds.get(name))

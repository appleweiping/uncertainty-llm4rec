from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShadowVariantSpec:
    variant: str
    method_name: str
    signal_role: str
    prompt_path: str
    primary_score_field: str
    support_fields: tuple[str, ...]
    output_fields: tuple[str, ...]
    score_formula: str
    uncertainty_formula: str
    lora_stage: str


SHADOW_VARIANTS: dict[str, ShadowVariantSpec] = {
    "shadow_v1": ShadowVariantSpec(
        variant="shadow_v1",
        method_name="C-CRP",
        signal_role="calibrated candidate relevance posterior",
        prompt_path="prompts/shadow_v1_relevance_probability.txt",
        primary_score_field="relevance_probability",
        support_fields=("evidence_support", "counterevidence_strength"),
        output_fields=(
            "relevance_probability",
            "evidence_support",
            "counterevidence_strength",
            "reason",
        ),
        score_formula="S_v1 = p_cal * (1 - U_v1)^eta",
        uncertainty_formula="U_v1 = alpha*U_boundary + beta*U_cal_gap + gamma*U_evidence",
        lora_stage="Signal LoRA first, Decision LoRA only after signal validation",
    ),
    "shadow_v2": ShadowVariantSpec(
        variant="shadow_v2",
        method_name="T-KIP",
        signal_role="top-k/frontier inclusion posterior",
        prompt_path="prompts/shadow_v2_topk_inclusion_probability.txt",
        primary_score_field="topk_inclusion_probability",
        support_fields=(
            "cutoff_margin_estimate",
            "competitive_pressure",
            "evidence_support",
            "counterevidence_strength",
        ),
        output_fields=(
            "topk_inclusion_probability",
            "cutoff_margin_estimate",
            "competitive_pressure",
            "evidence_support",
            "counterevidence_strength",
            "reason",
        ),
        score_formula="S_v2 = pi_i * (1 - U_v2)^eta",
        uncertainty_formula="U_v2 = alpha*U_cutoff + beta*U_cal_gap + gamma*U_budget + delta*U_comp",
        lora_stage="Listwise-aware Signal LoRA with candidate-set metadata",
    ),
    "shadow_v3": ShadowVariantSpec(
        variant="shadow_v3",
        method_name="U-PFS",
        signal_role="user-conditioned preference field strength",
        prompt_path="prompts/shadow_v3_preference_strength.txt",
        primary_score_field="preference_strength",
        support_fields=(
            "facet_alignment",
            "facet_conflict",
            "history_support",
            "novelty_pressure",
        ),
        output_fields=(
            "preference_strength",
            "facet_alignment",
            "facet_conflict",
            "history_support",
            "novelty_pressure",
            "reason",
        ),
        score_formula="S_v3 = preference_strength * facet_alignment * (1 - U_v3)^eta",
        uncertainty_formula="U_v3 combines facet conflict, weak history support, and novelty pressure",
        lora_stage="Facet-aware Signal LoRA before preference-pair construction",
    ),
    "shadow_v4": ShadowVariantSpec(
        variant="shadow_v4",
        method_name="RPD-ERU",
        signal_role="rank-position distribution with expected rank uncertainty",
        prompt_path="prompts/shadow_v4_rank_position_distribution.txt",
        primary_score_field="expected_rank_percentile",
        support_fields=(
            "rank_entropy",
            "frontier_probability",
            "rank_confidence",
        ),
        output_fields=(
            "expected_rank_percentile",
            "rank_entropy",
            "frontier_probability",
            "rank_confidence",
            "reason",
        ),
        score_formula="S_v4 = (1 - expected_rank_percentile) * frontier_probability * (1 - U_v4)^eta",
        uncertainty_formula="U_v4 combines rank entropy, frontier ambiguity, and low rank confidence",
        lora_stage="Rank-distribution Signal LoRA, then frontier-rerank adapter",
    ),
    "shadow_v5": ShadowVariantSpec(
        variant="shadow_v5",
        method_name="IGMP",
        signal_role="intent-prototype generation and match posterior",
        prompt_path="prompts/shadow_v5_intent_prototype_match.txt",
        primary_score_field="match_probability",
        support_fields=(
            "prototype_confidence",
            "match_evidence",
            "mismatch_strength",
        ),
        output_fields=(
            "intent_prototype",
            "match_probability",
            "prototype_confidence",
            "match_evidence",
            "mismatch_strength",
            "reason",
        ),
        score_formula="S_v5 = match_probability * prototype_confidence * (1 - U_v5)^eta",
        uncertainty_formula="U_v5 combines prototype uncertainty, match ambiguity, and mismatch strength",
        lora_stage="Prototype-and-match Signal LoRA with short controllable prototype targets",
    ),
    "shadow_v6": ShadowVariantSpec(
        variant="shadow_v6",
        method_name="SCARF",
        signal_role="signal-conditioned adaptive rerank and fine-tuning bridge",
        prompt_path="prompts/shadow_v6_signal_to_decision.txt",
        primary_score_field="decision_score",
        support_fields=(
            "signal_score",
            "signal_uncertainty",
            "correction_gate",
            "fallback_flag",
            "pair_weight",
        ),
        output_fields=(
            "decision_score",
            "signal_score",
            "signal_uncertainty",
            "correction_gate",
            "fallback_flag",
            "pair_type",
            "pair_weight",
            "reason",
        ),
        score_formula="S_v6 = gate * S_shadow + (1 - gate) * S_anchor",
        uncertainty_formula="U_v6 uses winner-signal uncertainty plus correction-risk and anchor disagreement",
        lora_stage="Decision LoRA bridge after selecting a winning shadow signal",
    ),
}


def normalize_shadow_variant(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized.startswith("v") and normalized[1:].isdigit():
        normalized = f"shadow_{normalized}"
    if normalized.isdigit():
        normalized = f"shadow_v{normalized}"
    return normalized


def get_shadow_variant(value: str) -> ShadowVariantSpec:
    variant = normalize_shadow_variant(value)
    if variant not in SHADOW_VARIANTS:
        known = ", ".join(sorted(SHADOW_VARIANTS))
        raise KeyError(f"Unknown shadow variant: {value}. Known variants: {known}")
    return SHADOW_VARIANTS[variant]

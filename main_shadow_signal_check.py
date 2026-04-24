from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.shadow import SHADOW_VARIANTS, compute_shadow_scores, parse_shadow_response


EXAMPLES = {
    "shadow_v1": {
        "relevance_probability": 0.72,
        "evidence_support": 0.64,
        "counterevidence_strength": 0.18,
        "reason": "The candidate matches several recent user interests.",
    },
    "shadow_v2": {
        "topk_inclusion_probability": 0.67,
        "cutoff_margin_estimate": 0.22,
        "competitive_pressure": 0.41,
        "evidence_support": 0.62,
        "counterevidence_strength": 0.2,
        "reason": "It is likely to sit near the front of the candidate set.",
    },
    "shadow_v3": {
        "preference_strength": 0.69,
        "facet_alignment": 0.7,
        "facet_conflict": 0.18,
        "history_support": 0.66,
        "novelty_pressure": 0.28,
        "reason": "The candidate aligns with a stable preference facet.",
    },
    "shadow_v4": {
        "expected_rank_percentile": 0.24,
        "rank_entropy": 0.32,
        "frontier_probability": 0.7,
        "rank_confidence": 0.68,
        "reason": "The expected rank is near the recommendation frontier.",
    },
    "shadow_v5": {
        "intent_prototype": "recent concise paperback fiction with broad appeal",
        "match_probability": 0.73,
        "prototype_confidence": 0.66,
        "match_evidence": 0.69,
        "mismatch_strength": 0.16,
        "reason": "The item matches the inferred intent prototype.",
    },
    "shadow_v6": {
        "decision_score": 0.71,
        "signal_score": 0.74,
        "signal_uncertainty": 0.22,
        "correction_gate": 0.63,
        "fallback_flag": False,
        "pair_type": "shadow_preferred_over_anchor",
        "pair_weight": 0.68,
        "reason": "The shadow signal is reliable enough to override the anchor.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="all", help="shadow_v1..shadow_v6, or all.")
    parser.add_argument("--check_prompts", action="store_true", help="Verify prompt template files exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = sorted(SHADOW_VARIANTS) if args.variant == "all" else [args.variant]
    for variant in variants:
        spec = SHADOW_VARIANTS[variant]
        payload = json.dumps(EXAMPLES[variant])
        parsed = parse_shadow_response(payload, variant=variant)
        scores = compute_shadow_scores(parsed, variant=variant)
        prompt_status = "not_checked"
        if args.check_prompts:
            prompt_status = "ready" if Path(spec.prompt_path).exists() else "missing"
        print(
            json.dumps(
                {
                    "variant": variant,
                    "method_name": spec.method_name,
                    "parse_success": parsed["parse_success"],
                    "prompt_status": prompt_status,
                    "primary_score": parsed["shadow_primary_score"],
                    **scores,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()

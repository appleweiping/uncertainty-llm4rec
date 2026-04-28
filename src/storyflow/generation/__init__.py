"""Generative recommendation prompt and output modules."""

from storyflow.generation.prompts import (
    CATALOG_CONSTRAINED_JSON_TEMPLATE,
    FORCED_JSON_TEMPLATE,
    NEXT_TITLE_TEMPLATE,
    PROBABILITY_CONFIDENCE_TEMPLATE,
    SELF_VERIFICATION_TEMPLATE,
    build_catalog_constrained_json_prompt,
    build_forced_json_prompt,
    build_next_title_prompt,
    build_probability_confidence_prompt,
    build_prompt,
    build_self_verification_prompt,
    compute_prompt_hash,
    format_candidate_titles,
    format_history_titles,
)

__all__ = [
    "CATALOG_CONSTRAINED_JSON_TEMPLATE",
    "FORCED_JSON_TEMPLATE",
    "NEXT_TITLE_TEMPLATE",
    "PROBABILITY_CONFIDENCE_TEMPLATE",
    "SELF_VERIFICATION_TEMPLATE",
    "build_catalog_constrained_json_prompt",
    "build_forced_json_prompt",
    "build_next_title_prompt",
    "build_probability_confidence_prompt",
    "build_prompt",
    "build_self_verification_prompt",
    "compute_prompt_hash",
    "format_candidate_titles",
    "format_history_titles",
]

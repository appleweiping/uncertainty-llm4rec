"""Generative recommendation prompt and output modules."""

from storyflow.generation.prompts import (
    FORCED_JSON_TEMPLATE,
    NEXT_TITLE_TEMPLATE,
    PROBABILITY_CONFIDENCE_TEMPLATE,
    SELF_VERIFICATION_TEMPLATE,
    build_forced_json_prompt,
    build_next_title_prompt,
    build_probability_confidence_prompt,
    build_prompt,
    build_self_verification_prompt,
    compute_prompt_hash,
    format_history_titles,
)

__all__ = [
    "FORCED_JSON_TEMPLATE",
    "NEXT_TITLE_TEMPLATE",
    "PROBABILITY_CONFIDENCE_TEMPLATE",
    "SELF_VERIFICATION_TEMPLATE",
    "build_forced_json_prompt",
    "build_next_title_prompt",
    "build_probability_confidence_prompt",
    "build_prompt",
    "build_self_verification_prompt",
    "compute_prompt_hash",
    "format_history_titles",
]

"""Prompt templates for title-level generative recommendation observation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A named prompt template for generative recommendation."""

    name: str
    description: str


NEXT_TITLE_TEMPLATE = PromptTemplate(
    name="next_title",
    description="Generate only the next item title from user history.",
)
SELF_VERIFICATION_TEMPLATE = PromptTemplate(
    name="next_title_self_verification",
    description="Generate the next title and answer yes/no self-verification.",
)
PROBABILITY_CONFIDENCE_TEMPLATE = PromptTemplate(
    name="next_title_probability_confidence",
    description="Generate the next title and report probability of correctness.",
)
FORCED_JSON_TEMPLATE = PromptTemplate(
    name="forced_json",
    description="Generate a JSON object with title, yes/no, and confidence.",
)
CATALOG_CONSTRAINED_JSON_TEMPLATE = PromptTemplate(
    name="catalog_constrained_json",
    description=(
        "Diagnostic prompt that asks for a JSON title selected from a provided "
        "catalog candidate list."
    ),
)


def format_history_titles(history_titles: Iterable[str]) -> str:
    """Format chronological title history for prompts."""

    titles = [str(title).strip() for title in history_titles if str(title).strip()]
    if not titles:
        raise ValueError("history_titles must contain at least one non-empty title")
    return "\n".join(f"{index + 1}. {title}" for index, title in enumerate(titles))


def format_candidate_titles(candidate_titles: Iterable[str]) -> str:
    """Format catalog candidate titles for diagnostic grounding prompts."""

    titles = [str(title).strip() for title in candidate_titles if str(title).strip()]
    if not titles:
        raise ValueError("candidate_titles must contain at least one non-empty title")
    return "\n".join(f"{index + 1}. {title}" for index, title in enumerate(titles))


def build_next_title_prompt(history_titles: Iterable[str]) -> str:
    """Prompt for free-form next-title generation."""

    history = format_history_titles(history_titles)
    return (
        "You are doing title-level generative recommendation.\n"
        "The user has interacted with these items in chronological order:\n"
        f"{history}\n\n"
        "Generate the single next item title the user is most likely to interact "
        "with. Return only the item title, not an item id and not a ranked list."
    )


def build_self_verification_prompt(history_titles: Iterable[str]) -> str:
    """Prompt for next-title generation with yes/no self-verification."""

    history = format_history_titles(history_titles)
    return (
        "You are doing title-level generative recommendation.\n"
        "The user history is a chronological list of item titles:\n"
        f"{history}\n\n"
        "Generate one next item title. Then answer whether your generated title "
        "is likely to be correct for this user's next interaction using Yes or No."
    )


def build_probability_confidence_prompt(history_titles: Iterable[str]) -> str:
    """Prompt for next-title generation with a probability confidence."""

    history = format_history_titles(history_titles)
    return (
        "You are doing title-level generative recommendation.\n"
        "The user history is a chronological list of item titles:\n"
        f"{history}\n\n"
        "Generate one next item title and report the probability that this "
        "recommendation is correct as a number in [0, 1]."
    )


def build_forced_json_prompt(history_titles: Iterable[str]) -> str:
    """Prompt that requires a parseable JSON response."""

    history = format_history_titles(history_titles)
    return (
        "You are doing title-level generative recommendation, not ranking from a "
        "shown candidate set.\n"
        "The user history is a chronological list of item titles:\n"
        f"{history}\n\n"
        "Generate exactly one next item title. Return only valid JSON with this "
        'schema: {"generated_title": string, "is_likely_correct": "yes" | '
        '"no", "confidence": number}. The confidence must be the probability '
        "in [0, 1] that the generated title is correct after catalog grounding."
    )


def build_catalog_constrained_json_prompt(
    history_titles: Iterable[str],
    candidate_titles: Iterable[str],
) -> str:
    """Diagnostic prompt for grounding-gate checks with catalog candidates."""

    history = format_history_titles(history_titles)
    candidates = format_candidate_titles(candidate_titles)
    return (
        "You are doing title-level generative recommendation.\n"
        "This is a catalog-grounding diagnostic prompt: choose exactly one "
        "groundable item title from the provided catalog candidate titles, "
        "based on the user's chronological history.\n\n"
        "User history:\n"
        f"{history}\n\n"
        "Catalog candidate titles:\n"
        f"{candidates}\n\n"
        "Return only valid JSON with this schema: "
        '{"generated_title": string, "is_likely_correct": "yes" | "no", '
        '"confidence": number}. The generated_title must exactly match one '
        "candidate title. If no candidate is plausible, set generated_title "
        'to "NO_GROUNDABLE_TITLE", is_likely_correct to "no", and confidence '
        "to 0.0. Confidence must be in [0, 1]."
    )


def build_prompt(
    history_titles: Iterable[str],
    *,
    template: str = FORCED_JSON_TEMPLATE.name,
    candidate_titles: Iterable[str] | None = None,
) -> str:
    """Build a prompt by template name."""

    if template == NEXT_TITLE_TEMPLATE.name:
        return build_next_title_prompt(history_titles)
    if template == SELF_VERIFICATION_TEMPLATE.name:
        return build_self_verification_prompt(history_titles)
    if template == PROBABILITY_CONFIDENCE_TEMPLATE.name:
        return build_probability_confidence_prompt(history_titles)
    if template == FORCED_JSON_TEMPLATE.name:
        return build_forced_json_prompt(history_titles)
    if template == CATALOG_CONSTRAINED_JSON_TEMPLATE.name:
        if candidate_titles is None:
            raise ValueError("candidate_titles are required for catalog_constrained_json")
        return build_catalog_constrained_json_prompt(history_titles, candidate_titles)
    raise ValueError(f"unknown prompt template: {template}")


def compute_prompt_hash(prompt: str) -> str:
    """Stable hash for prompt deduplication and resumable observation."""

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

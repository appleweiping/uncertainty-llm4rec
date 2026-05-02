"""Prompt builder with target-leakage safeguards."""

from __future__ import annotations

import json
from typing import Any

from llm4rec.data.text_fields import item_text
from llm4rec.prompts.base import PromptBuildResult, make_prompt_result
from llm4rec.prompts.templates import (
    CANDIDATE_NORMALIZED_TEMPLATE,
    CANDIDATE_NORMALIZED_TEMPLATE_ID,
    GENERATIVE_TITLE_TEMPLATE,
    GENERATIVE_TITLE_TEMPLATE_ID,
    RERANK_TEMPLATE,
    RERANK_TEMPLATE_ID,
    YES_NO_VERIFY_TEMPLATE,
    YES_NO_VERIFY_TEMPLATE_ID,
)

_LOOKUP_CACHE: dict[int, tuple[int, dict[str, dict[str, Any]]]] = {}


def build_generative_title_prompt(
    *,
    example: dict[str, Any],
    item_catalog: list[dict[str, Any]],
    candidate_items: list[str],
    text_policy: str = "title",
    exclude_item_ids: set[str] | None = None,
) -> PromptBuildResult:
    excluded = _target_exclusions(example, exclude_item_ids)
    context = _prompt_context(
        example=example,
        item_catalog=item_catalog,
        candidate_items=candidate_items,
        text_policy=text_policy,
        exclude_item_ids=excluded,
    )
    prompt = GENERATIVE_TITLE_TEMPLATE.format(
        history_titles=_json_list(context["history_titles"]),
        candidate_titles=_json_list(context["candidate_titles"]),
    )
    return make_prompt_result(
        prompt=prompt,
        prompt_template_id=GENERATIVE_TITLE_TEMPLATE_ID,
        metadata={**context, "task": "generative_title"},
    )


def build_rerank_prompt(
    *,
    example: dict[str, Any],
    item_catalog: list[dict[str, Any]],
    candidate_items: list[str],
    text_policy: str = "title",
    exclude_item_ids: set[str] | None = None,
) -> PromptBuildResult:
    excluded = _target_exclusions(example, exclude_item_ids)
    context = _prompt_context(
        example=example,
        item_catalog=item_catalog,
        candidate_items=candidate_items,
        text_policy=text_policy,
        exclude_item_ids=excluded,
    )
    prompt = RERANK_TEMPLATE.format(
        history_titles=_json_list(context["history_titles"]),
        candidate_titles=_json_list(context["candidate_titles"]),
    )
    return make_prompt_result(
        prompt=prompt,
        prompt_template_id=RERANK_TEMPLATE_ID,
        metadata={**context, "task": "candidate_rerank"},
    )


def build_yes_no_verification_prompt(
    *,
    example: dict[str, Any],
    item_catalog: list[dict[str, Any]],
    generated_title: str,
    grounded_title: str | None = None,
    text_policy: str = "title",
) -> PromptBuildResult:
    excluded = _target_exclusions(example, None)
    context = _prompt_context(
        example=example,
        item_catalog=item_catalog,
        candidate_items=[],
        text_policy=text_policy,
        exclude_item_ids=excluded,
    )
    prompt = YES_NO_VERIFY_TEMPLATE.format(
        history_titles=_json_list(context["history_titles"]),
        generated_title=generated_title,
        grounded_title=grounded_title or "",
    )
    return make_prompt_result(
        prompt=prompt,
        prompt_template_id=YES_NO_VERIFY_TEMPLATE_ID,
        metadata={
            **context,
            "task": "yes_no_verification",
            "generated_title": generated_title,
            "grounded_title": grounded_title,
        },
    )


def build_candidate_normalized_prompt(
    *,
    example: dict[str, Any],
    item_catalog: list[dict[str, Any]],
    generated_title: str,
    candidate_items: list[str],
    text_policy: str = "title",
    exclude_item_ids: set[str] | None = None,
) -> PromptBuildResult:
    excluded = _target_exclusions(example, exclude_item_ids)
    context = _prompt_context(
        example=example,
        item_catalog=item_catalog,
        candidate_items=candidate_items,
        text_policy=text_policy,
        exclude_item_ids=excluded,
    )
    prompt = CANDIDATE_NORMALIZED_TEMPLATE.format(
        history_titles=_json_list(context["history_titles"]),
        generated_title=generated_title,
        candidate_titles=_json_list(context["candidate_titles"]),
    )
    return make_prompt_result(
        prompt=prompt,
        prompt_template_id=CANDIDATE_NORMALIZED_TEMPLATE_ID,
        metadata={
            **context,
            "task": "candidate_normalized",
            "generated_title": generated_title,
        },
    )


def _prompt_context(
    *,
    example: dict[str, Any],
    item_catalog: list[dict[str, Any]],
    candidate_items: list[str],
    text_policy: str,
    exclude_item_ids: set[str] | None,
) -> dict[str, Any]:
    lookup = _catalog_lookup(item_catalog)
    excluded = {str(item_id) for item_id in (exclude_item_ids or set()) if str(item_id)}
    raw_history_ids = [str(item_id) for item_id in example.get("history", [])]
    prompt_history_ids = [
        item_id for item_id in raw_history_ids if item_id not in excluded and item_id in lookup
    ]
    prompt_candidate_ids = [
        str(item_id)
        for item_id in candidate_items
        if str(item_id) not in excluded and str(item_id) in lookup
    ]
    return {
        "example_id": example.get("example_id"),
        "user_id": str(example.get("user_id") or ""),
        "history_item_ids": prompt_history_ids,
        "history_titles": [_safe_item_text(lookup[item_id], text_policy) for item_id in prompt_history_ids],
        "history_excluded_item_ids": [item_id for item_id in raw_history_ids if item_id in excluded],
        "prompt_candidate_item_ids": prompt_candidate_ids,
        "candidate_titles": [_safe_item_text(lookup[item_id], text_policy) for item_id in prompt_candidate_ids],
        "excluded_item_ids": sorted(excluded),
        "target_excluded_from_prompt": str(example.get("target")) in excluded,
        "text_policy": text_policy,
    }


def _target_exclusions(example: dict[str, Any], exclude_item_ids: set[str] | None) -> set[str]:
    excluded = {str(item_id) for item_id in (exclude_item_ids or set()) if str(item_id)}
    target = str(example.get("target") or "")
    if target:
        excluded.add(target)
    return excluded


def _safe_item_text(row: dict[str, Any], policy: str) -> str:
    item = _item_record_like(row)
    return item_text(item, policy=policy)


def _item_record_like(row: dict[str, Any]) -> Any:
    class _Item:
        item_id = str(row.get("item_id") or "")
        title = str(row.get("title") or "")
        description = row.get("description")
        category = row.get("category") or row.get("genres")
        brand = row.get("brand")
        domain = row.get("domain")
        raw_text = row.get("raw_text")

    return _Item()


def _json_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _catalog_lookup(item_catalog: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cache_key = id(item_catalog)
    cached = _LOOKUP_CACHE.get(cache_key)
    if cached is not None and cached[0] == len(item_catalog):
        return cached[1]
    lookup = {str(row["item_id"]): row for row in item_catalog}
    _LOOKUP_CACHE[cache_key] = (len(item_catalog), lookup)
    return lookup

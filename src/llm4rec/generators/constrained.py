"""Helpers for constraining generated titles to candidate items."""

from __future__ import annotations

from typing import Any

from llm4rec.grounding.title import ground_title


def constrained_title_to_item(
    *,
    generated_title: str,
    item_catalog: list[dict[str, Any]],
    candidate_items: list[str],
) -> dict[str, Any]:
    result = ground_title(generated_title, item_catalog)
    candidate_set = {str(item_id) for item_id in candidate_items}
    metadata = result.to_dict()
    metadata["candidate_adherent"] = bool(
        result.grounding_success and result.grounded_item_id in candidate_set
    )
    return metadata

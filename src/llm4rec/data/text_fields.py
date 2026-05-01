"""Item text construction policies for prompts and text retrieval."""

from __future__ import annotations

from llm4rec.data.base import ItemRecord


def item_text(item: ItemRecord, *, policy: str = "title") -> str:
    """Return deterministic item text for a configured policy."""

    if policy == "title":
        return item.title
    if policy == "title_category":
        parts = [item.title]
        if item.category:
            parts.append(str(item.category))
        return " | ".join(parts)
    if policy == "title_description":
        parts = [item.title]
        if item.description:
            parts.append(str(item.description))
        return " ".join(parts)
    raise ValueError(f"unknown item text policy: {policy}")

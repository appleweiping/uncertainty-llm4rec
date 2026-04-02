# src/data/text_builder.py
from __future__ import annotations

from typing import Any

import pandas as pd


def clean_text_fields(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        flat = []
        for x in value:
            if isinstance(x, list):
                flat.extend(str(i) for i in x if i is not None)
            elif x is not None:
                flat.append(str(x))
        value = " | ".join(flat)
    text = str(value).strip()
    return " ".join(text.split())


def build_candidate_text(
    row: pd.Series,
    strategy: str = "title_categories_description",
    max_desc_len: int = 1000,
    fallback_to_title_categories: bool = True,
) -> str:
    title = clean_text_fields(row.get("title", ""))
    categories = clean_text_fields(row.get("categories", ""))
    description = clean_text_fields(row.get("description", ""))[:max_desc_len]

    if strategy == "title_categories_description":
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if categories:
            parts.append(f"Categories: {categories}")
        if description:
            parts.append(f"Description: {description}")

        if not description and fallback_to_title_categories:
            return "\n".join(p for p in parts if p)

        return "\n".join(p for p in parts if p)

    raise ValueError(f"Unknown text strategy: {strategy}")


def attach_candidate_text(
    items: pd.DataFrame,
    strategy: str = "title_categories_description",
    max_desc_len: int = 1000,
    fallback_to_title_categories: bool = True,
) -> pd.DataFrame:
    out = items.copy()
    out["candidate_text"] = out.apply(
        lambda row: build_candidate_text(
            row,
            strategy=strategy,
            max_desc_len=max_desc_len,
            fallback_to_title_categories=fallback_to_title_categories,
        ),
        axis=1,
    )
    return out
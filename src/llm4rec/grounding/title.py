"""Title-to-catalog grounding for generated recommendation titles."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TitleGroundingResult:
    generated_title: str
    grounded_item_id: str | None
    grounded_title: str | None
    grounding_score: float
    grounding_method: str
    grounding_success: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_title": self.generated_title,
            "grounded_item_id": self.grounded_item_id,
            "grounded_title": self.grounded_title,
            "grounding_score": self.grounding_score,
            "grounding_method": self.grounding_method,
            "grounding_success": self.grounding_success,
        }


def ground_title(
    generated_title: str,
    item_catalog: list[dict[str, Any]],
    *,
    min_token_overlap: float = 0.5,
) -> TitleGroundingResult:
    raw = str(generated_title or "").strip()
    if not raw:
        return _failed(raw, "empty")
    catalog = _sorted_catalog(item_catalog)
    for row in catalog:
        title = str(row.get("title") or "")
        if raw.casefold() == title.casefold():
            return _success(raw, row, 1.0, "exact_case_insensitive")
    normalized = normalize_title(raw)
    for row in catalog:
        title = str(row.get("title") or "")
        if normalized and normalized == normalize_title(title):
            return _success(raw, row, 0.98, "normalized")
    best_row: dict[str, Any] | None = None
    best_score = 0.0
    for row in catalog:
        score = token_overlap(raw, str(row.get("title") or ""))
        if score > best_score:
            best_score = score
            best_row = row
    if best_row is not None and best_score >= min_token_overlap:
        return _success(raw, best_row, best_score, "token_overlap")
    return _failed(raw, "no_match")


def normalize_title(title: str) -> str:
    table = str.maketrans({char: " " for char in string.punctuation})
    normalized = title.casefold().translate(table)
    return re.sub(r"\s+", " ", normalized).strip()


def token_overlap(left: str, right: str) -> float:
    left_tokens = set(normalize_title(left).split())
    right_tokens = set(normalize_title(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _sorted_catalog(item_catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        item_catalog,
        key=lambda row: (str(row.get("item_id") or ""), normalize_title(str(row.get("title") or ""))),
    )


def _success(raw: str, row: dict[str, Any], score: float, method: str) -> TitleGroundingResult:
    return TitleGroundingResult(
        generated_title=raw,
        grounded_item_id=str(row.get("item_id")),
        grounded_title=str(row.get("title")),
        grounding_score=float(score),
        grounding_method=method,
        grounding_success=True,
    )


def _failed(raw: str, method: str) -> TitleGroundingResult:
    return TitleGroundingResult(
        generated_title=raw,
        grounded_item_id=None,
        grounded_title=None,
        grounding_score=0.0,
        grounding_method=method,
        grounding_success=False,
    )

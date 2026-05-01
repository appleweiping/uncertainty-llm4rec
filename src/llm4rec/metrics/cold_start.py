"""Cold-start slicing helpers for evaluation only."""

from __future__ import annotations

from typing import Any


def user_history_bucket(history_length: int) -> str:
    length = int(history_length)
    if length <= 2:
        return "cold_user"
    if length <= 5:
        return "warm_user"
    return "heavy_user"


def metadata_availability_bucket(item: dict[str, Any] | None) -> str:
    if not item:
        return "metadata_missing"
    has_text = bool(item.get("title") or item.get("description") or item.get("raw_text"))
    has_category = bool(item.get("category") or item.get("brand") or item.get("domain"))
    return "metadata_available" if has_text or has_category else "metadata_missing"

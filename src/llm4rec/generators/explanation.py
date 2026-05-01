"""Explanation metadata scaffold for future LLM explanation baselines."""

from __future__ import annotations

from typing import Any


def explanation_record(*, reason: str | None, uncertainty_reason: str | None = None) -> dict[str, Any]:
    return {
        "reason": reason,
        "uncertainty_reason": uncertainty_reason,
        "is_phase3_scaffold": True,
    }

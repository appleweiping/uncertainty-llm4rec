"""Generator contracts for structured recommendation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class GenerationResult:
    generated_title: str | None
    confidence: float | None
    parse_success: bool
    raw_output: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseGenerator(Protocol):
    def generate(self, example: dict[str, Any], candidate_items: list[str]) -> GenerationResult:
        """Generate a structured recommendation output."""

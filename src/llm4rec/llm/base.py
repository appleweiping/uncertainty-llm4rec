"""LLM provider contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class LLMRequest:
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    raw_response: dict[str, Any]
    usage: dict[str, int]
    latency_seconds: float
    model: str
    provider: str
    cache_hit: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(Protocol):
    provider_name: str
    model_name: str
    supports_logprobs: bool
    supports_seed: bool

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one response for a prompt."""

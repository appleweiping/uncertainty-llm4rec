# src/llm/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class GenerationResult:
    raw_text: str
    latency: float
    model_name: str
    provider: str | None = None
    usage: dict[str, Any] | None = None
    raw_response: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["usage"] is None:
            data["usage"] = {}
        return data


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> dict[str, Any]:
        raise NotImplementedError


def normalize_generation_result(
    result: GenerationResult | dict[str, Any] | str,
    *,
    default_model_name: str = "unknown",
    default_provider: str | None = None,
) -> dict[str, Any]:
    if isinstance(result, GenerationResult):
        normalized = result.to_dict()
    elif isinstance(result, dict):
        normalized = {
            "raw_text": str(result.get("raw_text", result.get("text", "")) or ""),
            "latency": float(result.get("latency", 0.0) or 0.0),
            "model_name": str(result.get("model_name", default_model_name) or default_model_name),
            "provider": result.get("provider", default_provider),
            "usage": result.get("usage", {}) or {},
            "raw_response": result.get("raw_response"),
        }
    else:
        normalized = {
            "raw_text": str(result or ""),
            "latency": 0.0,
            "model_name": default_model_name,
            "provider": default_provider,
            "usage": {},
            "raw_response": None,
        }

    normalized["raw_text"] = str(normalized.get("raw_text", "") or "").strip()
    normalized["latency"] = float(normalized.get("latency", 0.0) or 0.0)
    normalized["model_name"] = str(normalized.get("model_name", default_model_name) or default_model_name)
    normalized["usage"] = normalized.get("usage", {}) or {}
    normalized["provider"] = normalized.get("provider", default_provider)
    return normalized

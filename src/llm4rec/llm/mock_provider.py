"""Deterministic mock LLM provider for smoke tests only."""

from __future__ import annotations

import json
from typing import Any

from llm4rec.llm.base import LLMRequest, LLMResponse


class MockLLMProvider:
    provider_name = "mock"
    model_name = "mock-llm"
    supports_logprobs = False
    supports_seed = True

    def __init__(self, *, response_mode: str = "generative_correct", seed: int = 0) -> None:
        self.response_mode = response_mode
        self.seed = int(seed)

    def generate(self, request: LLMRequest) -> LLMResponse:
        task = str(request.metadata.get("task") or "")
        mode = self.response_mode
        if mode == "malformed_json":
            text = "not valid json"
        elif task == "candidate_rerank" or mode.startswith("rerank_"):
            text = self._rerank_text(request, mode=mode)
        elif task == "yes_no_verification" or mode == "yes_no_verify":
            text = self._yes_no_text()
        elif task == "candidate_normalized" or mode == "candidate_normalized":
            text = self._candidate_normalized_text(request)
        else:
            text = self._generative_text(request, mode=mode)
        usage = _usage(request.prompt, text)
        return LLMResponse(
            text=text,
            raw_response={"text": text, "mode": mode, "task": task},
            usage=usage,
            latency_seconds=0.0,
            model=self.model_name,
            provider=self.provider_name,
            cache_hit=False,
            metadata={"response_mode": mode, "seed": self.seed},
        )

    def _generative_text(self, request: LLMRequest, *, mode: str) -> str:
        candidates = _candidate_titles(request)
        title = candidates[0] if candidates else "Alpha Movie"
        confidence = 0.82
        if mode == "generative_invalid":
            title = "Imaginary Catalog Item"
            confidence = 0.91
        elif mode == "generative_low_confidence":
            confidence = 0.2
        payload = {
            "recommendation": title,
            "confidence": confidence,
            "reason": "deterministic mock response",
            "uncertainty_reason": "mock uncertainty signal for smoke testing",
        }
        return json.dumps(payload, ensure_ascii=False)

    def _rerank_text(self, request: LLMRequest, *, mode: str) -> str:
        candidates = _candidate_titles(request)
        if mode == "rerank_reverse":
            ordered = list(reversed(candidates))
        else:
            ordered = list(candidates)
        payload = {
            "ranked_items": [
                {"title": title, "confidence": max(0.0, 0.9 - index * 0.1)}
                for index, title in enumerate(ordered)
            ],
            "reason": "deterministic mock reranking",
        }
        return json.dumps(payload, ensure_ascii=False)

    def _yes_no_text(self) -> str:
        return json.dumps(
            {"answer": "yes", "confidence": 0.74, "reason": "deterministic mock verification"},
            ensure_ascii=False,
        )

    def _candidate_normalized_text(self, request: LLMRequest) -> str:
        candidates = _candidate_titles(request)
        if not candidates:
            candidates = [str(request.metadata.get("generated_title") or "Alpha Movie")]
        weights = [1.0 / (index + 1) for index in range(len(candidates))]
        total = sum(weights) or 1.0
        payload = {
            "options": [
                {"title": title, "confidence": weight / total}
                for title, weight in zip(candidates, weights, strict=False)
            ],
            "normalized": True,
        }
        return json.dumps(payload, ensure_ascii=False)


def _candidate_titles(request: LLMRequest) -> list[str]:
    values = request.metadata.get("candidate_titles") or []
    return [str(value) for value in values if str(value)]


def _usage(prompt: str, text: str) -> dict[str, int]:
    prompt_tokens = len(prompt.split())
    completion_tokens = len(text.split())
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

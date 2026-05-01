"""OpenAI-compatible provider interface.

This adapter is intentionally not used by Phase 3 tests. It requires an API key
from the configured environment variable and performs no work until generate()
is called.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from llm4rec.llm.base import LLMRequest, LLMResponse


class OpenAICompatibleProvider:
    provider_name = "openai_compatible"
    supports_logprobs = False
    supports_seed = True

    def __init__(
        self,
        *,
        model_name: str,
        api_key_env: str,
        base_url: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.model_name = _required_text(model_name, "model_name")
        self.api_key_env = _required_text(api_key_env, "api_key_env")
        self.base_url = _required_text(base_url, "base_url").rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)

    def generate(self, request: LLMRequest) -> LLMResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key environment variable: {self.api_key_env}")
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        start = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:  # noqa: S310
                    raw = json.loads(response.read().decode("utf-8"))
                text = str(raw["choices"][0]["message"]["content"])
                usage = dict(raw.get("usage") or {})
                return LLMResponse(
                    text=text,
                    raw_response=_sanitize_raw(raw),
                    usage={key: int(value) for key, value in usage.items() if isinstance(value, int)},
                    latency_seconds=time.perf_counter() - start,
                    model=self.model_name,
                    provider=self.provider_name,
                    cache_hit=False,
                    metadata={"attempt": attempt, "api_key_env": self.api_key_env},
                )
            except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2.0, 0.25 * (2 ** attempt)))
        raise RuntimeError(f"OpenAI-compatible request failed: {last_error}") from last_error


def _sanitize_raw(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": raw.get("id"),
        "model": raw.get("model"),
        "usage": raw.get("usage"),
        "choices": raw.get("choices"),
    }


def _required_text(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    if not text or text.casefold() in {"none", "null"}:
        raise ValueError(f"{field_name} is required")
    return text

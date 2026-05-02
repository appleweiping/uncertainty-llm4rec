"""OpenAI-compatible provider interface.

This adapter is intentionally not used by Phase 3 tests. It requires an API key
from the configured environment variable and performs no work until generate()
is called.
"""

from __future__ import annotations

import json
import os
import ssl
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
        backoff_seconds: float = 0.25,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = _required_text(model_name, "model_name")
        self.api_key_env = _required_text(api_key_env, "api_key_env")
        self.base_url = _required_text(base_url, "base_url").rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.backoff_seconds = float(backoff_seconds)
        self.extra_body = _safe_extra_body(extra_body)
        self._ssl_context = _default_ssl_context()

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
        payload.update(self.extra_body)
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
                with urllib.request.urlopen(req, timeout=self.timeout_seconds, context=self._ssl_context) as response:  # noqa: S310
                    raw = json.loads(response.read().decode("utf-8"))
                message = raw["choices"][0]["message"]
                text = str(message.get("content") or "")
                usage = dict(raw.get("usage") or {})
                return LLMResponse(
                    text=text,
                    raw_response=_sanitize_raw(raw),
                    usage={key: int(value) for key, value in usage.items() if isinstance(value, int)},
                    latency_seconds=time.perf_counter() - start,
                    model=self.model_name,
                    provider=self.provider_name,
                    cache_hit=False,
                    metadata={
                        "attempt": attempt,
                        "api_key_env": self.api_key_env,
                        "extra_body_keys": sorted(self.extra_body),
                    },
                )
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.backoff_seconds * (2 ** attempt))
        raise RuntimeError(f"OpenAI-compatible request failed: {last_error}") from last_error


def _sanitize_raw(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": raw.get("id"),
        "model": raw.get("model"),
        "usage": raw.get("usage"),
        "choices": raw.get("choices"),
    }


def _default_ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _safe_extra_body(extra_body: dict[str, Any] | None) -> dict[str, Any]:
    if extra_body is None:
        return {}
    if not isinstance(extra_body, dict):
        raise ValueError("extra_body must be a mapping")
    blocked = {"model", "messages", "stream"}
    unsafe = sorted(key for key in extra_body if str(key) in blocked)
    if unsafe:
        raise ValueError(f"extra_body cannot override core payload fields: {unsafe}")
    return json.loads(json.dumps(extra_body))


def _required_text(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    if not text or text.casefold() in {"none", "null"}:
        raise ValueError(f"{field_name} is required")
    return text

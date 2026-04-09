from __future__ import annotations

import os
import time
from typing import Any

from openai import OpenAI

from src.llm.base import GenerationResult, LLMBackend


def read_api_key(api_key_env: str) -> str:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Required environment variable is not set: {api_key_env}")
    return api_key


def build_client_from_config(
    *,
    api_key_env: str,
    base_url: str | None = None,
    timeout: float | None = None,
) -> OpenAI:
    client_kwargs: dict[str, Any] = {
        "api_key": read_api_key(api_key_env),
    }
    if base_url:
        client_kwargs["base_url"] = base_url
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    return OpenAI(**client_kwargs)


def _normalize_usage(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {}


class APIBackend(LLMBackend):
    def __init__(
        self,
        *,
        provider: str,
        model_name: str,
        base_url: str | None,
        api_key_env: str,
        temperature: float = 0.0,
        max_tokens: int = 300,
        timeout: float | None = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self.extra_headers = extra_headers or {}
        self.client = build_client_from_config(
            api_key_env=api_key_env,
            base_url=base_url,
            timeout=timeout,
        )

    def call_api(self, prompt: str, **kwargs) -> Any:
        request_kwargs: dict[str, Any] = {
            "model": kwargs.get("model_name", self.model_name),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(kwargs.get("temperature", self.temperature)),
            "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
        }

        extra_body = kwargs.get("extra_body")
        merged_extra_body = dict(self.extra_body)
        if isinstance(extra_body, dict):
            merged_extra_body.update(extra_body)
        if merged_extra_body:
            request_kwargs["extra_body"] = merged_extra_body

        extra_headers = kwargs.get("extra_headers")
        merged_extra_headers = dict(self.extra_headers)
        if isinstance(extra_headers, dict):
            merged_extra_headers.update(extra_headers)
        if merged_extra_headers:
            request_kwargs["extra_headers"] = merged_extra_headers

        return self.client.chat.completions.create(**request_kwargs)

    def normalize_response(self, response: Any, *, latency: float) -> dict[str, Any]:
        content = ""
        model_name = self.model_name

        if response is not None:
            model_name = str(getattr(response, "model", None) or self.model_name)
            choices = getattr(response, "choices", None) or []
            if choices:
                message = getattr(choices[0], "message", None)
                content = getattr(message, "content", "") or ""

        result = GenerationResult(
            raw_text=str(content).strip(),
            latency=latency,
            model_name=model_name,
            provider=self.provider,
            usage=_normalize_usage(getattr(response, "usage", None)),
            raw_response=None,
        )
        return result.to_dict()

    def generate(self, prompt: str, **kwargs) -> dict[str, Any]:
        start = time.perf_counter()
        response = self.call_api(prompt, **kwargs)
        latency = time.perf_counter() - start
        return self.normalize_response(response, latency=latency)

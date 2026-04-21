from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI

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
        batch_size: int = 1,
        max_concurrency: int | None = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.batch_size = max(1, int(batch_size))
        resolved_concurrency = max_concurrency if max_concurrency is not None else self.batch_size
        self.max_concurrency = max(1, int(resolved_concurrency))
        self.extra_body = extra_body or {}
        self.extra_headers = extra_headers or {}
        self.max_retries = 8
        self.retry_backoff_seconds = 1.5
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

        max_retries = int(kwargs.get("max_retries", self.max_retries))
        retry_backoff_seconds = float(
            kwargs.get("retry_backoff_seconds", self.retry_backoff_seconds)
        )

        for attempt in range(max_retries + 1):
            try:
                return self.client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                is_retryable = (
                    status_code == 429
                    or (isinstance(status_code, int) and 500 <= status_code < 600)
                    or isinstance(exc, (APIConnectionError, APITimeoutError, TimeoutError))
                )
                if not is_retryable or attempt >= max_retries:
                    raise

                sleep_seconds = retry_backoff_seconds * (attempt + 1)
                time.sleep(sleep_seconds)

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

    def batch_generate(self, prompts: list[str], **kwargs) -> list[dict[str, Any]]:
        if len(prompts) <= 1 or self.max_concurrency <= 1:
            return [self.generate(prompt, **kwargs) for prompt in prompts]

        max_workers = min(self.max_concurrency, len(prompts))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(lambda prompt: self.generate(prompt, **kwargs), prompts))

"""OpenAI-compatible provider scaffold.

This module is intentionally conservative. It can execute only when the caller
explicitly disables dry-run, supplies an API key through the configured
environment variable, and confirms real endpoint/model settings in config.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from storyflow.providers.base import APIRequestRecord, APIResponseRecord, ProviderExecutionError
from storyflow.providers.config import ProviderConfig


class OpenAICompatibleProvider:
    """Minimal stdlib HTTP adapter for future real API pilots."""

    def __init__(self, config: ProviderConfig, *, api_key: str | None = None) -> None:
        self.config = config
        self.api_key = api_key if api_key is not None else os.environ.get(config.api_key_env)

    def _url(self) -> str:
        if self.config.requires_endpoint_confirmation or self.config.endpoint_is_placeholder:
            raise ProviderExecutionError(
                "Provider endpoint/model is a TODO placeholder; confirm config before real API execution."
            )
        base_url = (self.config.base_url or "").rstrip("/")
        endpoint = self.config.endpoint or ""
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        if not base_url or not endpoint:
            raise ProviderExecutionError("Provider base_url/endpoint is missing")
        return f"{base_url}/{endpoint.lstrip('/')}"

    def generate(self, request: APIRequestRecord, input_record: dict[str, object]) -> APIResponseRecord:
        if not self.api_key:
            raise ProviderExecutionError(f"Missing API key env var: {self.config.api_key_env}")
        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt,
                }
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url(),
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderExecutionError(str(exc)) from exc
        raw_text = _extract_openai_text(response_payload)
        return APIResponseRecord(
            request_id=request.request_id,
            input_id=request.input_id,
            provider=request.provider,
            model=request.model,
            raw_text=raw_text,
            status="ok",
            cache_key=request.cache_key,
            cache_hit=False,
            dry_run=False,
            usage=response_payload.get("usage", {}) if isinstance(response_payload, dict) else {},
        )


def _extract_openai_text(payload: dict[str, object]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and message.get("content") is not None:
                return str(message["content"])
            if first.get("text") is not None:
                return str(first["text"])
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)

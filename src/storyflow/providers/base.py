"""Base request/response records and provider interfaces."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from storyflow.providers.config import ProviderConfig


@dataclass(frozen=True, slots=True)
class APIRequestRecord:
    request_id: str
    input_id: str
    provider: str
    model: str
    prompt_template: str
    prompt_hash: str
    prompt: str
    cache_key: str
    temperature: float = 0.0
    max_tokens: int = 256
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class APIResponseRecord:
    request_id: str
    input_id: str
    provider: str
    model: str
    raw_text: str | None
    status: str
    cache_key: str
    cache_hit: bool
    dry_run: bool
    usage: dict[str, Any] = field(default_factory=dict)
    raw_payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at_unix: float = field(default_factory=time.time)


class ProviderExecutionError(RuntimeError):
    """Raised when a provider cannot execute a request safely."""


class APIProvider(Protocol):
    config: ProviderConfig

    def generate(self, request: APIRequestRecord, input_record: dict[str, Any]) -> APIResponseRecord:
        """Generate a provider response for one observation input."""


class DryRunProvider:
    """Deterministic no-network provider for API runner dry-runs."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.call_count = 0

    def generate(self, request: APIRequestRecord, input_record: dict[str, Any]) -> APIResponseRecord:
        self.call_count += 1
        payload = {
            "generated_title": input_record.get("target_title", "DRY RUN TITLE"),
            "is_likely_correct": "yes",
            "confidence": 0.5,
            "dry_run": True,
            "note": "Dry-run response; no network or paid API call was made.",
        }
        return APIResponseRecord(
            request_id=request.request_id,
            input_id=request.input_id,
            provider=request.provider,
            model=request.model,
            raw_text=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            status="ok",
            cache_key=request.cache_key,
            cache_hit=False,
            dry_run=True,
            usage={
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "estimated_cost": None,
            },
            raw_payload=payload,
        )

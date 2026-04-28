"""Provider configuration loading for API observation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from storyflow.utils.config import load_simple_yaml


@dataclass(frozen=True, slots=True)
class RetryConfig:
    max_attempts: int = 3
    backoff_seconds: float = 2.0


@dataclass(frozen=True, slots=True)
class RateLimitConfig:
    requests_per_minute: int = 20


@dataclass(frozen=True, slots=True)
class CacheConfig:
    enabled: bool = True
    cache_dir: str = "outputs/api_cache/default"


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    provider_name: str
    provider_family: str
    model_name: str
    api_key_env: str
    base_url: str | None
    endpoint: str | None
    requires_endpoint_confirmation: bool
    timeout_seconds: int
    max_concurrency: int
    temperature: float
    max_tokens: int
    retry: RetryConfig
    rate_limit: RateLimitConfig
    cache: CacheConfig
    dry_run_default: bool
    notes: str = ""

    @property
    def endpoint_is_placeholder(self) -> bool:
        values = [self.base_url or "", self.endpoint or "", self.model_name]
        return any("TODO_CONFIRM" in value for value in values)


def _nested(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    return value if isinstance(value, dict) else {}


def provider_config_from_dict(config: dict[str, Any]) -> ProviderConfig:
    retry = _nested(config, "retry")
    rate_limit = _nested(config, "rate_limit")
    cache = _nested(config, "cache")
    dry_run = _nested(config, "dry_run")
    return ProviderConfig(
        provider_name=str(config["provider_name"]),
        provider_family=str(config.get("provider_family") or "openai_compatible"),
        model_name=str(config.get("model_name") or "TODO_CONFIRM_MODEL"),
        api_key_env=str(config.get("api_key_env") or ""),
        base_url=config.get("base_url"),
        endpoint=config.get("endpoint"),
        requires_endpoint_confirmation=bool(config.get("requires_endpoint_confirmation", True)),
        timeout_seconds=int(config.get("timeout_seconds") or 60),
        max_concurrency=int(config.get("max_concurrency") or 1),
        temperature=float(config.get("temperature") or 0.0),
        max_tokens=int(config.get("max_tokens") or 256),
        retry=RetryConfig(
            max_attempts=int(retry.get("max_attempts") or 3),
            backoff_seconds=float(retry.get("backoff_seconds") or 2.0),
        ),
        rate_limit=RateLimitConfig(
            requests_per_minute=int(rate_limit.get("requests_per_minute") or 20),
        ),
        cache=CacheConfig(
            enabled=bool(cache.get("enabled", True)),
            cache_dir=str(cache.get("cache_dir") or "outputs/api_cache/default"),
        ),
        dry_run_default=bool(dry_run.get("default", True)),
        notes=str(config.get("notes") or ""),
    )


def load_provider_config(path: str | Path) -> ProviderConfig:
    return provider_config_from_dict(load_simple_yaml(path))

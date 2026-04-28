"""Readiness checks for tightly gated API observation pilots."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from storyflow.providers.config import ProviderConfig, load_provider_config


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _count_jsonl_records(path: Path, *, limit: int | None = None) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
                if limit is not None and count >= limit:
                    break
    return count


def _has_todo(value: str | None) -> bool:
    return "TODO_CONFIRM" in str(value or "")


def check_api_pilot_readiness(
    *,
    provider_config_path: str | Path,
    input_jsonl: str | Path,
    sample_size: int,
    stage: str = "smoke",
    approved_provider: str | None = None,
    approved_model: str | None = None,
    approved_rate_limit: int | None = None,
    approved_max_concurrency: int | None = None,
    approved_budget_label: str | None = None,
    execute_api_intended: bool = False,
    allow_over_20: bool = False,
) -> dict[str, Any]:
    """Check whether a future real API pilot is allowed by project gates.

    This function never calls the network and never prints API key values. It
    only checks whether the configured key environment variable is present.
    """

    provider_config_path = Path(provider_config_path)
    input_jsonl = Path(input_jsonl)
    config: ProviderConfig = load_provider_config(provider_config_path)
    blockers: list[str] = []
    warnings: list[str] = []

    if stage not in {"smoke", "pilot"}:
        blockers.append("stage must be smoke or pilot")
    max_allowed = 5 if stage == "smoke" else 20
    if sample_size < 1:
        blockers.append("sample_size must be >= 1")
    if sample_size > max_allowed and not (stage == "pilot" and allow_over_20):
        blockers.append(f"{stage} sample_size must be <= {max_allowed}")
    if stage == "pilot" and allow_over_20 and sample_size > max_allowed:
        warnings.append(
            "pilot sample_size exceeds the default <=20 gate because allow_over_20 was explicitly set"
        )
    if approved_provider and approved_provider != config.provider_name:
        blockers.append(
            f"approved_provider={approved_provider} does not match config provider={config.provider_name}"
        )
    if not approved_provider:
        blockers.append("approved_provider is required before real API execution")
    if approved_model and approved_model != config.model_name:
        blockers.append(
            f"approved_model={approved_model} does not match config model={config.model_name}"
        )
    if not approved_model:
        blockers.append("approved_model is required before real API execution")
    if approved_rate_limit is None or approved_rate_limit < 1:
        blockers.append("approved_rate_limit >= 1 is required before real API execution")
    if approved_max_concurrency is None:
        approved_max_concurrency = config.max_concurrency
    if approved_max_concurrency < 1:
        blockers.append("approved_max_concurrency >= 1 is required before real API execution")
    if not approved_budget_label:
        blockers.append("approved_budget_label is required before real API execution")
    if not execute_api_intended:
        blockers.append("future real API command must explicitly include --execute-api")

    if _has_todo(config.model_name) or _has_todo(config.base_url) or _has_todo(config.endpoint):
        blockers.append("provider config still contains TODO_CONFIRM placeholder")
    if config.requires_endpoint_confirmation:
        blockers.append("provider config requires endpoint/model confirmation")
    if not config.base_url or not config.endpoint:
        blockers.append("provider base_url/endpoint is missing")
    env_present = bool(os.environ.get(config.api_key_env))
    if not env_present:
        blockers.append(f"environment variable {config.api_key_env} is not present")
    if not input_jsonl.exists():
        blockers.append(f"input_jsonl does not exist: {input_jsonl}")
        input_count = 0
    else:
        input_count = _count_jsonl_records(input_jsonl)
        if input_count < sample_size:
            blockers.append(
                f"input_jsonl has {input_count} records, fewer than requested sample_size={sample_size}"
            )
    if approved_max_concurrency != 1 and stage == "smoke":
        warnings.append("initial smoke should use max_concurrency=1 unless explicitly approved")
    effective_rate = approved_rate_limit or config.rate_limit.requests_per_minute
    if effective_rate > 10 and stage == "smoke":
        warnings.append("recommended smoke-test rate_limit is <= 10 requests/minute")
    if not config.cache.enabled:
        blockers.append("provider cache must be enabled before real API execution")

    status = "ready_for_execute_api" if not blockers else "blocked"
    command_template = (
        "python scripts/run_api_observation.py "
        f"--provider-config {provider_config_path} "
        f"--input-jsonl {input_jsonl} "
        f"--max-examples {sample_size} "
        "--execute-api "
        f"--rate-limit {effective_rate} "
        f"--max-concurrency {approved_max_concurrency} "
        f"--budget-label {approved_budget_label or 'APPROVED_BUDGET_LABEL'}"
    )
    return {
        "created_at_utc": utc_now_iso(),
        "status": status,
        "stage": stage,
        "provider_config": str(provider_config_path),
        "provider": config.provider_name,
        "model": config.model_name,
        "base_url": config.base_url,
        "endpoint": config.endpoint,
        "api_key_env": config.api_key_env,
        "api_key_env_present": env_present,
        "api_key_value_printed": False,
        "input_jsonl": str(input_jsonl),
        "input_count": input_count,
        "sample_size": sample_size,
        "approved_provider": approved_provider,
        "approved_model": approved_model,
        "approved_rate_limit": approved_rate_limit,
        "approved_max_concurrency": approved_max_concurrency,
        "approved_budget_label": approved_budget_label,
        "allow_over_20": allow_over_20,
        "execute_api_intended": execute_api_intended,
        "cache_enabled": config.cache.enabled,
        "cache_dir": config.cache.cache_dir,
        "blockers": blockers,
        "warnings": warnings,
        "dry_run_only": True,
        "api_called": False,
        "command_template_after_approval": command_template,
        "note": "Readiness check only. It does not call DeepSeek or any paid API.",
    }

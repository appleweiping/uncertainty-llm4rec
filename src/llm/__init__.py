from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.llm.api_backend import APIBackend
from src.llm.deepseek_backend import DeepSeekBackend
from src.llm.openai_backend import OpenAIBackend


BACKEND_REGISTRY = {
    "api": APIBackend,
    "openai_compatible": APIBackend,
    "deepseek": DeepSeekBackend,
    "openai": OpenAIBackend,
}


def _first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def load_model_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_backend_from_config(model_cfg_path: str | Path):
    model_cfg = load_model_config(model_cfg_path)
    backend_name = str(model_cfg.get("backend_name", "")).strip().lower()
    if not backend_name:
        raise ValueError(f"backend_name is required in model config: {model_cfg_path}")

    backend_cls = BACKEND_REGISTRY.get(backend_name)
    if backend_cls is None:
        raise ValueError(f"Unsupported backend_name: {backend_name}")

    provider = str(
        _first_present(
            model_cfg.get("provider"),
            model_cfg.get("provider_name"),
            backend_name,
        )
    ).strip().lower()

    generation_cfg = model_cfg.get("generation", {}) or {}
    connection_cfg = model_cfg.get("connection", {}) or {}

    model_name = _first_present(model_cfg.get("model_name"), generation_cfg.get("model_name"))
    if not model_name:
        raise ValueError(f"model_name is required in model config: {model_cfg_path}")

    temperature = float(_first_present(generation_cfg.get("temperature"), model_cfg.get("temperature"), 0.0))
    max_tokens = int(_first_present(generation_cfg.get("max_tokens"), model_cfg.get("max_tokens"), 300))
    base_url = _first_present(connection_cfg.get("base_url"), model_cfg.get("base_url"))
    api_key_env = str(_first_present(connection_cfg.get("api_key_env"), model_cfg.get("api_key_env"), "") or "").strip()
    timeout = _first_present(connection_cfg.get("timeout"), model_cfg.get("timeout"))

    if not api_key_env:
        raise ValueError(f"api_key_env is required in model config: {model_cfg_path}")

    if backend_name in {"api", "openai_compatible"}:
        return backend_cls(
            provider=provider,
            model_name=str(model_name),
            base_url=str(base_url) if base_url is not None else None,
            api_key_env=api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=float(timeout) if timeout is not None else None,
            extra_body=model_cfg.get("extra_body"),
            extra_headers=model_cfg.get("extra_headers"),
        )

    if backend_name in {"deepseek", "openai"}:
        return backend_cls(
            model_name=str(model_name),
            base_url=str(base_url) if base_url is not None else None,
            api_key_env=api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=float(timeout) if timeout is not None else None,
        )

    raise ValueError(f"Unsupported backend_name: {backend_name}")

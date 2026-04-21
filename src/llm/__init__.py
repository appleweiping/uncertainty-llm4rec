from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.llm.api_backend import APIBackend
from src.llm.deepseek_backend import DeepSeekBackend
from src.llm.local_hf_backend import LocalHFBackend
from src.llm.openai_backend import OpenAIBackend


BACKEND_REGISTRY = {
    "api": APIBackend,
    "openai_compatible": APIBackend,
    "deepseek": DeepSeekBackend,
    "hf": LocalHFBackend,
    "local_hf": LocalHFBackend,
    "transformers": LocalHFBackend,
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


def build_backend_from_dict(model_cfg: dict[str, Any]):
    backend_name = str(model_cfg.get("backend_name", "")).strip().lower()
    if not backend_name:
        raise ValueError("backend_name is required in model config.")

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
    runtime_cfg = model_cfg.get("runtime", {}) or {}

    if backend_name in {"hf", "local_hf", "transformers"}:
        model_name_or_path = _first_present(
            model_cfg.get("model_name_or_path"),
            runtime_cfg.get("model_name_or_path"),
            model_cfg.get("model_path"),
            model_cfg.get("model_name"),
        )
        if not model_name_or_path:
            raise ValueError("model_name_or_path is required in local HF model config.")
        tokenizer_name_or_path = _first_present(
            model_cfg.get("tokenizer_name_or_path"),
            runtime_cfg.get("tokenizer_name_or_path"),
            model_cfg.get("tokenizer_path"),
            model_name_or_path,
        )
        return backend_cls(
            provider=provider,
            model_name=str(_first_present(model_cfg.get("model_name"), Path(str(model_name_or_path)).name)),
            model_name_or_path=str(model_name_or_path),
            tokenizer_name_or_path=str(tokenizer_name_or_path),
            device=str(_first_present(runtime_cfg.get("device"), model_cfg.get("device"), "cuda")),
            device_map=_first_present(runtime_cfg.get("device_map"), model_cfg.get("device_map"), "auto"),
            dtype=str(_first_present(runtime_cfg.get("dtype"), model_cfg.get("dtype"), "auto")),
            batch_size=int(_first_present(runtime_cfg.get("batch_size"), model_cfg.get("batch_size"), 1)),
            max_new_tokens=int(_first_present(generation_cfg.get("max_new_tokens"), generation_cfg.get("max_tokens"), model_cfg.get("max_new_tokens"), model_cfg.get("max_tokens"), 300)),
            temperature=float(_first_present(generation_cfg.get("temperature"), model_cfg.get("temperature"), 0.0)),
            top_p=float(_first_present(generation_cfg.get("top_p"), model_cfg.get("top_p"), 1.0)),
            trust_remote_code=bool(_first_present(runtime_cfg.get("trust_remote_code"), model_cfg.get("trust_remote_code"), False)),
            local_files_only=bool(_first_present(runtime_cfg.get("local_files_only"), model_cfg.get("local_files_only"), True)),
            load_in_4bit=bool(_first_present(runtime_cfg.get("load_in_4bit"), model_cfg.get("load_in_4bit"), False)),
            load_in_8bit=bool(_first_present(runtime_cfg.get("load_in_8bit"), model_cfg.get("load_in_8bit"), False)),
            adapter_path=_first_present(runtime_cfg.get("adapter_path"), model_cfg.get("adapter_path")),
            use_chat_template=bool(_first_present(generation_cfg.get("use_chat_template"), model_cfg.get("use_chat_template"), True)),
            enable_thinking=_first_present(generation_cfg.get("enable_thinking"), model_cfg.get("enable_thinking")),
        )

    model_name = _first_present(model_cfg.get("model_name"), generation_cfg.get("model_name"))
    if not model_name:
        raise ValueError("model_name is required in model config.")

    temperature = float(_first_present(generation_cfg.get("temperature"), model_cfg.get("temperature"), 0.0))
    max_tokens = int(_first_present(generation_cfg.get("max_tokens"), model_cfg.get("max_tokens"), 300))
    base_url = _first_present(connection_cfg.get("base_url"), model_cfg.get("base_url"))
    api_key_env = str(_first_present(connection_cfg.get("api_key_env"), model_cfg.get("api_key_env"), "") or "").strip()
    timeout = _first_present(connection_cfg.get("timeout"), model_cfg.get("timeout"))
    batch_size = int(_first_present(runtime_cfg.get("batch_size"), model_cfg.get("batch_size"), 1))
    max_concurrency = _first_present(runtime_cfg.get("max_concurrency"), model_cfg.get("max_concurrency"))

    if not api_key_env:
        raise ValueError("api_key_env is required in model config.")

    if backend_name in {"api", "openai_compatible"}:
        return backend_cls(
            provider=provider,
            model_name=str(model_name),
            base_url=str(base_url) if base_url is not None else None,
            api_key_env=api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=float(timeout) if timeout is not None else None,
            batch_size=batch_size,
            max_concurrency=int(max_concurrency) if max_concurrency is not None else None,
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
            batch_size=batch_size,
            max_concurrency=int(max_concurrency) if max_concurrency is not None else None,
        )

    raise ValueError(f"Unsupported backend_name: {backend_name}")


def build_backend_from_config(model_cfg_path: str | Path):
    model_cfg = load_model_config(model_cfg_path)
    return build_backend_from_dict(model_cfg)

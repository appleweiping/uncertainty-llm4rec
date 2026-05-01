from __future__ import annotations

import builtins

import pytest

from llm4rec.llm.base import LLMRequest
from llm4rec.llm.hf_provider import HFLocalProvider
from llm4rec.llm.openai_provider import OpenAICompatibleProvider
from llm4rec.experiments.runner import _build_llm_provider


def test_openai_compatible_provider_requires_explicit_configuration() -> None:
    with pytest.raises(ValueError, match="model_name is required"):
        OpenAICompatibleProvider(model_name="", api_key_env="OPENAI_API_KEY", base_url="https://example.test")
    with pytest.raises(ValueError, match="api_key_env is required"):
        OpenAICompatibleProvider(model_name="model", api_key_env="", base_url="https://example.test")
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatibleProvider(model_name="model", api_key_env="OPENAI_API_KEY", base_url="")


def test_openai_compatible_provider_missing_api_key_fails_before_network(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAICompatibleProvider(
        model_name="model",
        api_key_env="TRUCE_REC_TEST_MISSING_KEY",
        base_url="https://example.test",
    )
    monkeypatch.delenv("TRUCE_REC_TEST_MISSING_KEY", raising=False)
    with pytest.raises(RuntimeError, match="missing API key environment variable"):
        provider.generate(LLMRequest(prompt="hello"))


def test_hf_provider_does_not_import_optional_dependencies_until_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = HFLocalProvider(model_name_or_path="local-model")
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name in {"torch", "transformers"}:
            raise ImportError(f"no {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="requires optional dependencies torch and transformers"):
        provider.generate(LLMRequest(prompt="hello"))


def test_hf_provider_requires_explicit_local_model_path() -> None:
    with pytest.raises(ValueError, match="model_name_or_path is required"):
        HFLocalProvider(model_name_or_path="")


def test_runner_rejects_null_real_provider_model_fields() -> None:
    with pytest.raises(ValueError, match="config.llm.model is required"):
        _build_llm_provider(
            {
                "llm": {
                    "provider": "openai_compatible",
                    "model": None,
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url": "https://example.test",
                }
            }
        )
    with pytest.raises(ValueError, match="config.llm.model_name_or_path is required"):
        _build_llm_provider({"llm": {"provider": "hf_local", "model_name_or_path": None}})

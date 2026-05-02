from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

from llm4rec.llm.base import LLMRequest
from llm4rec.llm.hf_provider import HFLocalProvider
from llm4rec.llm.openai_provider import OpenAICompatibleProvider
from llm4rec.experiments import runner as runner_module
from llm4rec.experiments.runner import _build_llm_provider, _cost_latency_for_run


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


def test_openai_compatible_provider_passes_safe_extra_body(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [{"message": {"content": "{\"ok\": true}"}}],
                    "model": "model",
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
                }
            ).encode("utf-8")

    def fake_urlopen(req: object, **_kwargs: object) -> FakeResponse:
        captured["payload"] = json.loads(req.data.decode("utf-8"))  # type: ignore[attr-defined]
        return FakeResponse()

    monkeypatch.setenv("TRUCE_REC_TEST_KEY", "not-a-real-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        model_name="model",
        api_key_env="TRUCE_REC_TEST_KEY",
        base_url="https://example.test",
        extra_body={"response_format": {"type": "json_object"}, "thinking": {"type": "disabled"}},
    )

    response = provider.generate(LLMRequest(prompt="hello", max_tokens=8))

    assert response.text == "{\"ok\": true}"
    assert captured["payload"] == {
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 8,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }


def test_openai_compatible_provider_rejects_core_extra_body_overrides() -> None:
    with pytest.raises(ValueError, match="extra_body cannot override"):
        OpenAICompatibleProvider(
            model_name="model",
            api_key_env="OPENAI_API_KEY",
            base_url="https://example.test",
            extra_body={"messages": []},
        )


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


def test_cache_only_replay_does_not_initialize_real_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_init(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("real provider must not be initialized in cache-only replay")

    monkeypatch.setattr(runner_module, "OpenAICompatibleProvider", fail_init)
    provider = _build_llm_provider(
        {
            "llm": {
                "provider": "openai_compatible",
                "model": "deepseek-v4-flash",
                "base_url": "https://api.deepseek.com",
                "api_key_env": "DEEPSEEK_API_KEY",
                "cache": {"enabled": True, "require_hit": True, "cache_dir": str(tmp_path)},
            },
            "safety": {"allow_api_calls": False},
        },
        run_dir=tmp_path,
    )

    with pytest.raises(RuntimeError, match="cache-only replay missing response cache entry"):
        provider.generate(LLMRequest(prompt="missing"))


def test_cache_only_replay_uses_cached_cost_latency_without_live_call(tmp_path: Path) -> None:
    provider = _build_llm_provider(
        {
            "llm": {
                "provider": "openai_compatible",
                "model": "deepseek-v4-flash",
                "base_url": "https://api.deepseek.com",
                "api_key_env": "DEEPSEEK_API_KEY",
                "cache": {"enabled": True, "require_hit": True, "cache_dir": str(tmp_path)},
                "pricing": {"input_per_1m_tokens": 0.14, "output_per_1m_tokens": 0.28},
            },
            "safety": {"allow_api_calls": False},
        },
        run_dir=tmp_path,
    )
    request = LLMRequest(prompt="hello", max_tokens=8)
    cache_key = provider._cache_key(request)
    provider.cache.set(
        cache_key,
        {
            "text": "{\"ok\": true}",
            "raw_response": {},
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "latency_seconds": 1.25,
            "model": "deepseek-v4-flash",
            "provider": "openai_compatible",
            "metadata": {"estimated_cost": 0.0000028},
        },
    )

    response = provider.generate(request)
    summary = provider.summary()

    assert response.cache_hit is True
    assert response.metadata["original_estimated_cost_usd"] == 0.0000028
    assert response.metadata["original_latency_seconds"] == 1.25
    assert summary["live_provider_requests"] == 0
    assert summary["cache_hit_requests"] == 1
    assert summary["original_cached_cost_usd"] == 0.0000028
    assert summary["effective_cost_usd"] == 0.0000028
    assert summary["original_cached_latency_seconds_sum"] == 1.25


def test_cost_latency_for_run_exposes_cache_replay_accounting(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "llm_provider_summary.json").write_text(
        json.dumps(
            {
                "request_count": 2,
                "real_request_count": 0,
                "cache_hit_count": 2,
                "cache_hit_rate": 1.0,
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
                "live_cost_usd": 0.0,
                "replay_cost_usd": 0.0,
                "original_cached_cost_usd": 0.004,
                "effective_cost_usd": 0.004,
                "replay_latency_seconds_sum": 0.01,
                "original_cached_latency_seconds_sum": 3.0,
                "latency_p50_seconds": 0.005,
                "original_live_latency_p50_seconds": 1.5,
            }
        ),
        encoding="utf-8",
    )

    cost_latency = _cost_latency_for_run({"aggregate": {"efficiency": {}}}, tmp_path)

    assert cost_latency["total_requests"] == 2
    assert cost_latency["live_provider_requests"] == 0
    assert cost_latency["cache_hit_requests"] == 2
    assert cost_latency["original_cached_cost_usd"] == 0.004
    assert cost_latency["effective_cost_usd"] == 0.004
    assert cost_latency["replay_latency_seconds_sum"] == 0.01
    assert cost_latency["original_cached_latency_seconds_sum"] == 3.0

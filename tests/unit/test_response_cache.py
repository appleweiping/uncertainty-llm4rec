from __future__ import annotations

import pytest

from llm4rec.experiments.runner import _response_from_cache
from llm4rec.llm.response_cache import ResponseCache


def test_response_cache_round_trip(tmp_path) -> None:
    cache = ResponseCache(tmp_path)
    key = cache.key_for(provider="mock", model="mock-llm", prompt="hello", params={"temperature": 0})
    assert cache.get(key) is None
    cache.set(key, {"text": "world"})
    assert cache.get(key) == {"text": "world"}


def test_response_cache_get_required_fails_on_missing_key(tmp_path) -> None:
    cache = ResponseCache(tmp_path)
    with pytest.raises(KeyError, match="missing response cache entry"):
        cache.get_required("missing")


def test_cached_response_preserves_original_usage_cost_and_latency() -> None:
    response = _response_from_cache(
        {
            "text": "{\"ok\": true}",
            "raw_response": {},
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "prompt_cache_hit_tokens": 5,
                "prompt_cache_miss_tokens": 6,
            },
            "latency_seconds": 2.5,
            "model": "deepseek-v4-flash",
            "provider": "openai_compatible",
            "metadata": {"estimated_cost": 0.0012},
        },
        cache_key="abc123",
        replay_latency_seconds=0.004,
    )

    assert response.cache_hit is True
    assert response.usage["prompt_tokens"] == 11
    assert response.usage["prompt_cache_hit_tokens"] == 5
    assert response.latency_seconds == 0.004
    assert response.metadata["cache_key"] == "abc123"
    assert response.metadata["source"] == "cache"
    assert response.metadata["estimated_cost"] == 0.0
    assert response.metadata["replay_estimated_cost_usd"] == 0.0
    assert response.metadata["original_estimated_cost_usd"] == 0.0012
    assert response.metadata["original_latency_seconds"] == 2.5
    assert response.metadata["replay_latency_seconds"] == 0.004

from __future__ import annotations

from llm4rec.llm.response_cache import ResponseCache


def test_response_cache_round_trip(tmp_path) -> None:
    cache = ResponseCache(tmp_path)
    key = cache.key_for(provider="mock", model="mock-llm", prompt="hello", params={"temperature": 0})
    assert cache.get(key) is None
    cache.set(key, {"text": "world"})
    assert cache.get(key) == {"text": "world"}

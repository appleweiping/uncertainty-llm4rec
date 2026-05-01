from __future__ import annotations

from llm4rec.llm.base import LLMRequest
from llm4rec.llm.mock_provider import MockLLMProvider


def test_mock_provider_generates_deterministic_recommendation() -> None:
    provider = MockLLMProvider(response_mode="generative_correct", seed=13)
    request = LLMRequest(prompt="prompt", metadata={"task": "generative_title", "candidate_titles": ["Beta Movie"]})
    first = provider.generate(request)
    second = provider.generate(request)
    assert first.text == second.text
    assert "Beta Movie" in first.text
    assert first.usage["total_tokens"] > 0


def test_mock_provider_supports_rerank_and_malformed_modes() -> None:
    rerank = MockLLMProvider(response_mode="rerank_reverse")
    request = LLMRequest(
        prompt="prompt",
        metadata={"task": "candidate_rerank", "candidate_titles": ["Alpha Movie", "Beta Movie"]},
    )
    assert "Beta Movie" in rerank.generate(request).text
    malformed = MockLLMProvider(response_mode="malformed_json")
    assert malformed.generate(request).text == "not valid json"

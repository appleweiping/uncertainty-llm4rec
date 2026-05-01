from __future__ import annotations

from llm4rec.experiments.config import load_config
from llm4rec.llm.base import LLMRequest
from llm4rec.methods.fallback import build_fallback_router
from llm4rec.methods.ours_method import OursMethodRanker

from tests.unit.test_ours_method import EXAMPLE, ITEMS, TRAIN


class NoCallProvider:
    provider_name = "mock"
    model_name = "mock-llm"
    supports_logprobs = False
    supports_seed = True

    def generate(self, request: LLMRequest):  # pragma: no cover - should never run
        raise AssertionError("fallback-only ablation must not call the LLM provider")


def test_fallback_router_supports_required_methods() -> None:
    for method in ["bm25", "popularity", "sequential_markov"]:
        router = build_fallback_router(method)
        router.fit(TRAIN, ITEMS)
        result = router.rank(EXAMPLE, ["i4", "i5"])
        assert result.predicted_items
        assert set(result.predicted_items) == {"i4", "i5"}


def test_fallback_only_ablation_skips_llm_generation() -> None:
    ranker = OursMethodRanker(
        provider=NoCallProvider(),
        method_config=load_config("configs/methods/ours_fallback_only.yaml"),
        seed=13,
    )
    ranker.fit(TRAIN, ITEMS)
    result = ranker.rank(EXAMPLE, ["i4", "i5"])
    assert result.method == "ours_fallback_only"
    assert result.metadata["uncertainty_decision"] == "fallback"
    assert result.metadata["fallback_used"] is True
    assert result.metadata["parse_success"] is False
    assert set(result.predicted_items) == {"i4", "i5"}

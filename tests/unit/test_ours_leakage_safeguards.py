from __future__ import annotations

from llm4rec.experiments.config import load_config
from llm4rec.llm.base import LLMRequest
from llm4rec.llm.mock_provider import MockLLMProvider
from llm4rec.methods.ours_method import OursMethodRanker

from tests.unit.test_ours_method import EXAMPLE, ITEMS, TRAIN


class CapturingMockProvider(MockLLMProvider):
    def __init__(self) -> None:
        super().__init__(response_mode="generative_correct", seed=13)
        self.prompts: list[str] = []

    def generate(self, request: LLMRequest):
        self.prompts.append(request.prompt)
        return super().generate(request)


def test_ours_prompts_exclude_target_id_and_title() -> None:
    provider = CapturingMockProvider()
    ranker = OursMethodRanker(
        provider=provider,
        method_config=load_config("configs/methods/ours_uncertainty_guided.yaml"),
        seed=13,
    )
    ranker.fit(TRAIN, ITEMS)
    result = ranker.rank(EXAMPLE, ["i4", "i5"])
    assert provider.prompts
    for prompt in provider.prompts:
        assert "i4" not in prompt
        assert "Delta Movie" not in prompt
    assert result.metadata["target_excluded_from_prompt"] is True
    assert "i4" in result.metadata["excluded_item_ids"]


def test_confidence_policy_metadata_does_not_drive_from_target_label() -> None:
    provider = CapturingMockProvider()
    ranker = OursMethodRanker(
        provider=provider,
        method_config=load_config("configs/methods/ours_uncertainty_guided.yaml"),
        seed=13,
    )
    ranker.fit(TRAIN, ITEMS)
    first = ranker.rank({**EXAMPLE, "target": "i4"}, ["i4", "i5"])
    second = ranker.rank({**EXAMPLE, "target": "i3"}, ["i3", "i5"])
    assert first.metadata["uncertainty_decision"] == second.metadata["uncertainty_decision"]
    assert first.metadata["confidence"] == second.metadata["confidence"]

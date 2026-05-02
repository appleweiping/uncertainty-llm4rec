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


def test_numeric_target_id_does_not_false_positive_inside_allowed_title() -> None:
    provider = CapturingMockProvider()
    ranker = OursMethodRanker(
        provider=provider,
        method_config=load_config("configs/methods/ours_uncertainty_guided.yaml"),
        seed=13,
    )
    items = [
        {"item_id": "10", "title": "Allowed History"},
        {"item_id": "60", "title": "Held Out Target"},
        {"item_id": "101", "title": "Gone in 60 Seconds (2000)"},
    ]
    train = [
        {
            "example_id": "u1:1",
            "user_id": "u1",
            "history": ["10"],
            "target": "101",
            "candidates": ["60", "101"],
            "split": "train",
            "domain": "movies",
            "metadata": {},
        }
    ]
    example = {
        "example_id": "u1:2",
        "user_id": "u1",
        "history": ["10"],
        "target": "60",
        "candidates": ["60", "101"],
        "split": "test",
        "domain": "movies",
        "metadata": {},
    }
    ranker.fit(train, items)
    result = ranker.rank(example, ["60", "101"])
    assert result.metadata["target_excluded_from_prompt"] is True
    assert result.metadata["prompt_candidate_item_ids"] == ["101"]
    assert "Gone in 60 Seconds (2000)" in provider.prompts[0]
    assert "Held Out Target" not in provider.prompts[0]


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

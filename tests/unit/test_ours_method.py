from __future__ import annotations

from llm4rec.experiments.config import load_config
from llm4rec.llm.mock_provider import MockLLMProvider
from llm4rec.methods.ours_method import OursMethodRanker


ITEMS = [
    {"item_id": "i1", "title": "Alpha Movie", "category": "Drama", "domain": "tiny"},
    {"item_id": "i2", "title": "Beta Movie", "category": "Comedy", "domain": "tiny"},
    {"item_id": "i3", "title": "Gamma Movie", "category": "Sci-Fi", "domain": "tiny"},
    {"item_id": "i4", "title": "Delta Movie", "category": "Drama", "domain": "tiny"},
    {"item_id": "i5", "title": "Epsilon Movie", "category": "Action", "domain": "tiny"},
]

TRAIN = [
    {"user_id": "u1", "history": ["i2"], "target": "i1", "split": "train", "domain": "tiny"},
    {"user_id": "u2", "history": ["i1"], "target": "i2", "split": "train", "domain": "tiny"},
]

EXAMPLE = {
    "user_id": "u1",
    "history": ["i2", "i1", "i3"],
    "target": "i4",
    "candidates": ["i4", "i5"],
    "split": "test",
    "domain": "tiny",
    "example_id": "u1:3",
}


def test_ours_method_accepts_grounded_high_confidence_mock_generation() -> None:
    ranker = OursMethodRanker(
        provider=MockLLMProvider(response_mode="generative_correct", seed=13),
        method_config=load_config("configs/methods/ours_uncertainty_guided.yaml"),
        seed=13,
    )
    ranker.fit(TRAIN, ITEMS)
    result = ranker.rank(EXAMPLE, ["i4", "i5"])
    metadata = result.metadata
    assert result.method == "ours_uncertainty_guided"
    assert result.predicted_items == ["i5"]
    assert metadata["ours_method"] is True
    assert metadata["generated_title"] == "Epsilon Movie"
    assert metadata["confidence"] == 0.82
    assert metadata["grounding_success"] is True
    assert metadata["grounded_item_id"] == "i5"
    assert metadata["uncertainty_decision"] == "accept"
    assert metadata["fallback_method"] == "bm25"
    assert metadata["ablation_variant"] == "full"
    assert metadata["disabled_components"] == []
    assert metadata["parse_success"] is True
    assert metadata["target_excluded_from_prompt"] is True
    assert metadata["candidate_normalized_confidence"] == 1.0


def test_ours_method_low_confidence_uses_configured_fallback() -> None:
    ranker = OursMethodRanker(
        provider=MockLLMProvider(response_mode="generative_low_confidence", seed=13),
        method_config=load_config("configs/methods/ours_uncertainty_guided.yaml"),
        seed=13,
    )
    ranker.fit(TRAIN, ITEMS)
    result = ranker.rank(EXAMPLE, ["i4", "i5"])
    assert result.metadata["uncertainty_decision"] == "fallback"
    assert result.metadata["fallback_used"] is True
    assert result.metadata["confidence"] == 0.2
    assert set(result.predicted_items) == {"i4", "i5"}

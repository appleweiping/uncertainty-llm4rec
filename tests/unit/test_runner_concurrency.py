from __future__ import annotations

from typing import Any

from llm4rec.experiments.runner import _rank_eval_examples
from llm4rec.rankers.base import RankingResult


class SuccessfulLLMStyleRanker:
    def rank(self, example: dict[str, Any], candidate_items: list[str]) -> RankingResult:
        return RankingResult(
            user_id=str(example["user_id"]),
            target_item=str(example["target"]),
            candidate_items=[str(item) for item in candidate_items],
            predicted_items=[str(candidate_items[0])],
            scores=[1.0],
            method="llm_test",
            domain="tiny",
            raw_output="{\"recommendation\": \"ok\"}",
            metadata={"parse_success": True},
        )


def test_llm_concurrency_successes_are_not_replaced_with_api_failures(tmp_path) -> None:
    examples = [
        {
            "user_id": f"u{index}",
            "target": "i1",
            "candidates": ["i1", "i2"],
            "domain": "tiny",
            "split": "test",
            "example_id": f"ex-{index}",
        }
        for index in range(3)
    ]
    config = {
        "method": {"name": "llm_generative"},
        "llm": {"provider": "openai_compatible"},
        "safety": {"concurrency": 4},
    }

    predictions = _rank_eval_examples(
        config,
        ranker=SuccessfulLLMStyleRanker(),
        eval_examples=examples,
        trainer_metadata={},
        run_dir=tmp_path,
    )

    assert [row["predicted_items"] for row in predictions] == [["i1"], ["i1"], ["i1"]]
    assert all(not row["metadata"].get("api_failure") for row in predictions)
    assert not (tmp_path / "artifacts" / "api_failures.jsonl").exists()

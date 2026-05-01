"""LLM candidate reranking baseline."""

from __future__ import annotations

from typing import Any

from llm4rec.grounding.title import ground_title
from llm4rec.llm.base import BaseLLMProvider, LLMRequest
from llm4rec.prompts.builder import build_rerank_prompt
from llm4rec.prompts.parsers import parse_rerank_response
from llm4rec.rankers.base import CheckpointNotImplementedMixin, RankingResult
from llm4rec.rankers.llm_generative import _response_metadata, _train_popularity


class LLMReranker(CheckpointNotImplementedMixin):
    method_name = "llm_rerank_mock"

    def __init__(
        self,
        *,
        provider: BaseLLMProvider,
        method_name: str = "llm_rerank_mock",
        text_policy: str = "title",
    ) -> None:
        self.provider = provider
        self.method_name = method_name
        self.text_policy = text_policy
        self.item_catalog: list[dict[str, Any]] = []
        self.popularity: dict[str, int] = {}

    def fit(
        self,
        train_examples: list[dict[str, Any]],
        item_catalog: list[dict[str, Any]],
        interactions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.item_catalog = item_catalog
        self.popularity = _train_popularity(train_examples)

    def rank(self, example: dict[str, Any], candidate_items: list[str]) -> RankingResult:
        target_id = str(example["target"])
        prompt = build_rerank_prompt(
            example=example,
            item_catalog=self.item_catalog,
            candidate_items=candidate_items,
            text_policy=self.text_policy,
            exclude_item_ids={target_id},
        )
        prompt_candidate_ids = [str(item_id) for item_id in prompt.metadata.get("prompt_candidate_item_ids", [])]
        response = self.provider.generate(LLMRequest(prompt=prompt.prompt, metadata=prompt.metadata))
        parsed = parse_rerank_response(response.text)
        ranked_titles = parsed.data.get("ranked_items", []) if parsed.parse_success else []
        ordered: list[str] = []
        scores: list[float] = []
        grounding_events: list[dict[str, Any]] = []
        for index, row in enumerate(ranked_titles):
            grounding = ground_title(str(row.get("title") or ""), self.item_catalog)
            grounding_event = grounding.to_dict()
            grounding_event["rank_index"] = index
            grounding_events.append(grounding_event)
            grounded_id = grounding.grounded_item_id
            if grounding.grounding_success and grounded_id in prompt_candidate_ids and grounded_id not in ordered:
                ordered.append(str(grounded_id))
                confidence = row.get("confidence")
                scores.append(float(confidence) if isinstance(confidence, (int, float)) else 0.0)
        for item_id in prompt_candidate_ids:
            if item_id not in ordered:
                ordered.append(item_id)
                scores.append(0.0)
        top_grounding = _candidate_grounding(ordered[0] if ordered else None, self.item_catalog)
        metadata = {
            "example_id": example.get("example_id"),
            "split": example.get("split"),
            "phase": "phase3_llm_baseline_observation",
            "not_ours_method": True,
            "parse_success": parsed.parse_success,
            "parse_error": parsed.error,
            "prompt_candidate_item_ids": prompt_candidate_ids,
            "excluded_item_ids": prompt.metadata.get("excluded_item_ids", []),
            "target_excluded_from_prompt": prompt.metadata.get("target_excluded_from_prompt"),
            "prompt_template_id": prompt.prompt_template_id,
            "prompt_hash": prompt.prompt_hash,
            "confidence": scores[0] if scores else None,
            "is_grounded_hit": bool(ordered and ordered[0] == target_id),
            "is_catalog_valid": bool(top_grounding["grounding_success"]),
            "is_hallucinated": not bool(top_grounding["grounding_success"]),
            "candidate_adherent": bool(ordered and ordered[0] in prompt_candidate_ids),
            "raw_ranked_grounding": grounding_events,
            **top_grounding,
            **_response_metadata(response),
        }
        return RankingResult(
            user_id=str(example["user_id"]),
            target_item=target_id,
            candidate_items=[str(item_id) for item_id in candidate_items],
            predicted_items=ordered,
            scores=scores,
            method=self.method_name,
            domain=str(example.get("domain") or "tiny"),
            raw_output=response.text,
            metadata=metadata,
        )


def _candidate_grounding(item_id: str | None, item_catalog: list[dict[str, Any]]) -> dict[str, Any]:
    if not item_id:
        return {
            "generated_title": "",
            "grounded_item_id": None,
            "grounded_title": None,
            "grounding_score": 0.0,
            "grounding_method": "empty",
            "grounding_success": False,
        }
    lookup = {str(row.get("item_id")): row for row in item_catalog}
    row = lookup.get(str(item_id))
    if row is None:
        return {
            "generated_title": str(item_id),
            "grounded_item_id": None,
            "grounded_title": None,
            "grounding_score": 0.0,
            "grounding_method": "missing_catalog_item",
            "grounding_success": False,
        }
    return {
        "generated_title": str(row.get("title") or ""),
        "grounded_item_id": str(item_id),
        "grounded_title": str(row.get("title") or ""),
        "grounding_score": 1.0,
        "grounding_method": "candidate_item_id",
        "grounding_success": True,
    }

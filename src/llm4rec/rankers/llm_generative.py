"""LLM generative recommendation and confidence-observation baselines."""

from __future__ import annotations

from typing import Any

from llm4rec.grounding.title import ground_title, token_overlap
from llm4rec.llm.base import BaseLLMProvider, LLMRequest
from llm4rec.prompts.builder import (
    build_candidate_normalized_prompt,
    build_generative_title_prompt,
    build_yes_no_verification_prompt,
)
from llm4rec.prompts.parsers import (
    parse_candidate_normalized_response,
    parse_generation_response,
    parse_yes_no_response,
)
from llm4rec.rankers.base import CheckpointNotImplementedMixin, RankingResult


class LLMGenerativeRanker(CheckpointNotImplementedMixin):
    method_name = "llm_generative_mock"

    def __init__(
        self,
        *,
        provider: BaseLLMProvider,
        method_name: str = "llm_generative_mock",
        text_policy: str = "title",
        observe_confidence: bool = False,
    ) -> None:
        self.provider = provider
        self.method_name = method_name
        self.text_policy = text_policy
        self.observe_confidence = observe_confidence
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
        prompt = build_generative_title_prompt(
            example=example,
            item_catalog=self.item_catalog,
            candidate_items=candidate_items,
            text_policy=self.text_policy,
            exclude_item_ids={target_id},
        )
        response = self.provider.generate(
            LLMRequest(prompt=prompt.prompt, metadata=prompt.metadata, seed=prompt.metadata.get("seed"))
        )
        parsed = parse_generation_response(response.text)
        generated_title = str(parsed.data.get("recommendation") or "") if parsed.parse_success else ""
        confidence = parsed.data.get("confidence") if parsed.parse_success else None
        grounding = ground_title(generated_title, self.item_catalog)
        predicted_items = [grounding.grounded_item_id] if grounding.grounding_success else [generated_title or "__parse_failed__"]
        scores = [float(confidence) if isinstance(confidence, (int, float)) else 0.0]
        metadata = self._metadata(
            example=example,
            prompt_metadata=prompt.metadata,
            response_metadata=_response_metadata(response),
            parsed_success=parsed.parse_success,
            parse_error=parsed.error,
            generated_title=generated_title,
            confidence=confidence,
            grounding=grounding.to_dict(),
            candidate_items=candidate_items,
        )
        if self.observe_confidence:
            metadata.update(self._confidence_observation(example, candidate_items, generated_title, grounding.to_dict()))
        return RankingResult(
            user_id=str(example["user_id"]),
            target_item=target_id,
            candidate_items=[str(item_id) for item_id in candidate_items],
            predicted_items=[str(item_id) for item_id in predicted_items],
            scores=scores,
            method=self.method_name,
            domain=str(example.get("domain") or "tiny"),
            raw_output=response.text,
            metadata=metadata,
        )

    def _confidence_observation(
        self,
        example: dict[str, Any],
        candidate_items: list[str],
        generated_title: str,
        grounding: dict[str, Any],
    ) -> dict[str, Any]:
        verify_prompt = build_yes_no_verification_prompt(
            example=example,
            item_catalog=self.item_catalog,
            generated_title=generated_title,
            grounded_title=grounding.get("grounded_title"),
            text_policy=self.text_policy,
        )
        verify_response = self.provider.generate(LLMRequest(prompt=verify_prompt.prompt, metadata=verify_prompt.metadata))
        verify_parsed = parse_yes_no_response(verify_response.text)
        normalized_prompt = build_candidate_normalized_prompt(
            example=example,
            item_catalog=self.item_catalog,
            generated_title=generated_title,
            candidate_items=candidate_items,
            text_policy=self.text_policy,
            exclude_item_ids={str(example["target"])},
        )
        normalized_response = self.provider.generate(
            LLMRequest(prompt=normalized_prompt.prompt, metadata=normalized_prompt.metadata)
        )
        normalized_parsed = parse_candidate_normalized_response(normalized_response.text)
        return {
            "verification_raw_output": verify_response.text,
            "verification_parse_success": verify_parsed.parse_success,
            "verification_answer": verify_parsed.data.get("answer") if verify_parsed.parse_success else None,
            "verification_confidence": verify_parsed.data.get("confidence") if verify_parsed.parse_success else None,
            "verification_prompt_template_id": verify_prompt.prompt_template_id,
            "verification_prompt_hash": verify_prompt.prompt_hash,
            "candidate_normalized_raw_output": normalized_response.text,
            "candidate_normalized_parse_success": normalized_parsed.parse_success,
            "candidate_normalized_options": normalized_parsed.data.get("options") if normalized_parsed.parse_success else [],
            "candidate_normalized_prompt_template_id": normalized_prompt.prompt_template_id,
            "candidate_normalized_prompt_hash": normalized_prompt.prompt_hash,
        }

    def _metadata(
        self,
        *,
        example: dict[str, Any],
        prompt_metadata: dict[str, Any],
        response_metadata: dict[str, Any],
        parsed_success: bool,
        parse_error: str | None,
        generated_title: str,
        confidence: Any,
        grounding: dict[str, Any],
        candidate_items: list[str],
    ) -> dict[str, Any]:
        grounded_id = grounding.get("grounded_item_id")
        metadata = {
            "example_id": example.get("example_id"),
            "split": example.get("split"),
            "phase": "phase3_llm_baseline_observation",
            "not_ours_method": True,
            "generated_title": generated_title,
            "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
            "parse_success": parsed_success,
            "parse_error": parse_error,
            "is_catalog_valid": bool(grounding.get("grounding_success")),
            "is_hallucinated": not bool(grounding.get("grounding_success")),
            "is_grounded_hit": bool(grounded_id and grounded_id == str(example["target"])),
            "candidate_adherent": bool(grounded_id and grounded_id in {str(item) for item in candidate_items}),
            **grounding,
            **prompt_metadata,
            **response_metadata,
            **self._popularity_metadata(grounded_id),
            **self._echo_metadata(example, grounded_id, generated_title),
        }
        return metadata

    def _popularity_metadata(self, item_id: Any) -> dict[str, Any]:
        if not item_id:
            return {"item_popularity": 0, "popularity_bucket": None}
        item_id = str(item_id)
        count = self.popularity.get(item_id, 0)
        positive = sorted({value for value in self.popularity.values() if value > 0})
        if not positive or count <= 0:
            bucket = "tail"
        elif count >= positive[-1]:
            bucket = "head"
        elif count <= positive[0]:
            bucket = "tail"
        else:
            bucket = "mid"
        return {"item_popularity": count, "popularity_bucket": bucket}

    def _echo_metadata(self, example: dict[str, Any], grounded_id: Any, generated_title: str) -> dict[str, Any]:
        lookup = {str(row["item_id"]): row for row in self.item_catalog}
        history_ids = [str(item_id) for item_id in example.get("history", [])]
        history_titles = [str(lookup[item_id].get("title") or "") for item_id in history_ids if item_id in lookup]
        history_categories = {
            str(lookup[item_id].get("category") or lookup[item_id].get("genres") or "")
            for item_id in history_ids
            if item_id in lookup
        }
        grounded_category = (
            str(lookup[str(grounded_id)].get("category") or lookup[str(grounded_id)].get("genres") or "")
            if grounded_id and str(grounded_id) in lookup
            else ""
        )
        history_similarity = max((token_overlap(generated_title, title) for title in history_titles), default=0.0)
        return {
            "history_title_overlap": history_similarity,
            "history_category_overlap": bool(grounded_category and grounded_category in history_categories),
            "generated_item_seen_in_history": bool(grounded_id and str(grounded_id) in set(history_ids)),
            "confidence_weighted_history_similarity": history_similarity,
        }


class LLMConfidenceObservationRanker(LLMGenerativeRanker):
    method_name = "llm_confidence_observation_mock"

    def __init__(
        self,
        *,
        provider: BaseLLMProvider,
        method_name: str = "llm_confidence_observation_mock",
        text_policy: str = "title",
    ) -> None:
        super().__init__(
            provider=provider,
            method_name=method_name,
            text_policy=text_policy,
            observe_confidence=True,
        )


def _train_popularity(train_examples: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in train_examples:
        for item_id in [*example.get("history", []), example.get("target")]:
            if item_id:
                item_id = str(item_id)
                counts[item_id] = counts.get(item_id, 0) + 1
    return counts


def _response_metadata(response: Any) -> dict[str, Any]:
    return {
        "provider": response.provider,
        "model": response.model,
        "latency_seconds": response.latency_seconds,
        "token_usage": dict(response.usage),
        "cache_hit": response.cache_hit,
        "provider_metadata": dict(response.metadata),
    }

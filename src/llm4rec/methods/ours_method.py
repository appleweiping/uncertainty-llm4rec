"""Minimal Phase 6 OursMethod integration."""

from __future__ import annotations

from collections import Counter
from typing import Any

from llm4rec.grounding.title import ground_title, normalize_title, token_overlap
from llm4rec.llm.base import BaseLLMProvider, LLMRequest
from llm4rec.methods.ablation import AblationSettings, resolve_ablation_settings
from llm4rec.methods.fallback import FallbackRouter, build_fallback_router
from llm4rec.methods.uncertainty_policy import PolicyDecision, UncertaintyPolicy
from llm4rec.prompts.builder import build_candidate_normalized_prompt, build_generative_title_prompt
from llm4rec.prompts.parsers import parse_candidate_normalized_response, parse_generation_response
from llm4rec.rankers.base import CheckpointNotImplementedMixin, RankingResult


class OursMethodRanker(CheckpointNotImplementedMixin):
    """Generative title method with grounding, uncertainty routing, and fallback."""

    method_name = "ours_uncertainty_guided"

    def __init__(
        self,
        *,
        provider: BaseLLMProvider,
        method_config: dict[str, Any],
        seed: int = 0,
    ) -> None:
        self.provider = provider
        self.method_config = method_config
        self.params = dict(method_config.get("params") or {})
        self.method_name = str(self.params.get("prediction_method") or method_config.get("name") or self.method_name)
        self.text_policy = str(self.params.get("text_policy") or "title")
        self.seed = int(method_config.get("seed") or seed)
        self.ablation = resolve_ablation_settings(method_config)
        self.policy = UncertaintyPolicy(
            thresholds=dict(self.params.get("thresholds") or {}),
            policy=dict(self.params.get("policy") or {}),
            components=self.ablation.components,
        )
        fallback_params = dict(self.params.get("fallback_params") or {})
        self.fallback: FallbackRouter = build_fallback_router(
            str(self.params.get("fallback_method") or "bm25"),
            fallback_params,
        )
        self.item_catalog: list[dict[str, Any]] = []
        self.lookup: dict[str, dict[str, Any]] = {}
        self.train_popularity: Counter[str] = Counter()

    def fit(
        self,
        train_examples: list[dict[str, Any]],
        item_catalog: list[dict[str, Any]],
        interactions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.item_catalog = item_catalog
        self.lookup = {str(row["item_id"]): row for row in item_catalog}
        self.train_popularity = _train_popularity(train_examples)
        self.fallback.fit(train_examples, item_catalog, interactions)

    def rank(self, example: dict[str, Any], candidate_items: list[str]) -> RankingResult:
        candidates = [str(item_id) for item_id in candidate_items]
        if self.ablation.enabled("fallback_only"):
            decision = PolicyDecision("fallback", ["fallback_only_ablation"], {"echo_risk": False})
            return self._fallback_prediction(
                example=example,
                candidate_items=candidates,
                decision=decision,
                base_metadata=self._disabled_generation_metadata(example),
                raw_output=None,
            )

        target_id = str(example["target"])
        prompt = build_generative_title_prompt(
            example=example,
            item_catalog=self.item_catalog,
            candidate_items=candidates,
            text_policy=self.text_policy,
            exclude_item_ids={target_id},
        )
        self._assert_no_target_leakage(prompt.prompt, example, prompt.metadata)
        response = self.provider.generate(
            LLMRequest(prompt=prompt.prompt, metadata=prompt.metadata, seed=self.seed)
        )
        parsed = parse_generation_response(response.text)
        generated_title = str(parsed.data.get("recommendation") or "") if parsed.parse_success else ""
        confidence = _confidence_or_zero(parsed.data.get("confidence") if parsed.parse_success else None)
        grounding = ground_title(
            generated_title,
            self.item_catalog,
            min_token_overlap=float(self.params.get("min_grounding_token_overlap") or 0.5),
        )
        grounded_id = grounding.grounded_item_id
        candidate_normalized = self._candidate_normalized_confidence(
            example=example,
            candidate_items=candidates,
            generated_title=generated_title,
            grounded_title=grounding.grounded_title,
        )
        popularity = self._popularity_metadata(grounded_id)
        history_similarity = self._history_similarity(example, generated_title, grounded_id)
        signals = {
            "parse_success": parsed.parse_success,
            "confidence": confidence,
            "grounding_success": grounding.grounding_success,
            "grounding_score": grounding.grounding_score,
            "candidate_normalized_confidence": candidate_normalized.get("candidate_normalized_confidence"),
            "popularity_bucket": popularity["popularity_bucket"],
            "history_similarity": history_similarity,
            "grounded_item_in_candidates": bool(grounded_id and grounded_id in set(candidates)),
        }
        decision = self.policy.decide(signals)
        base_metadata = self._metadata(
            example=example,
            prompt_metadata=prompt.metadata,
            response_metadata=_response_metadata(response),
            parsed_success=parsed.parse_success,
            parse_error=parsed.error,
            generated_title=generated_title,
            confidence=confidence,
            grounding=grounding.to_dict(),
            candidate_normalized=candidate_normalized,
            popularity=popularity,
            history_similarity=history_similarity,
            decision=decision,
        )
        if decision.decision in {"fallback", "rerank"}:
            return self._fallback_prediction(
                example=example,
                candidate_items=candidates,
                decision=decision,
                base_metadata=base_metadata,
                raw_output=response.text,
            )
        if decision.decision == "abstain":
            return self._abstain_prediction(example, candidates, base_metadata, raw_output=response.text)
        return self._accept_prediction(
            example=example,
            candidate_items=candidates,
            generated_title=generated_title,
            grounded_id=grounded_id,
            confidence=confidence,
            base_metadata=base_metadata,
            raw_output=response.text,
        )

    def _candidate_normalized_confidence(
        self,
        *,
        example: dict[str, Any],
        candidate_items: list[str],
        generated_title: str,
        grounded_title: str | None,
    ) -> dict[str, Any]:
        if not self.ablation.enabled("candidate_normalized_confidence"):
            return {
                "candidate_normalized_confidence": None,
                "candidate_normalized_parse_success": None,
                "candidate_normalized_options": [],
                "candidate_normalized_prompt_template_id": None,
                "candidate_normalized_prompt_hash": None,
                "candidate_normalized_raw_output": None,
            }
        prompt = build_candidate_normalized_prompt(
            example=example,
            item_catalog=self.item_catalog,
            generated_title=generated_title,
            candidate_items=candidate_items,
            text_policy=self.text_policy,
            exclude_item_ids={str(example["target"])},
        )
        self._assert_no_target_leakage(prompt.prompt, example, prompt.metadata)
        response = self.provider.generate(LLMRequest(prompt=prompt.prompt, metadata=prompt.metadata, seed=self.seed))
        parsed = parse_candidate_normalized_response(response.text)
        options = parsed.data.get("options") if parsed.parse_success else []
        normalized = _matched_option_confidence(options, generated_title, grounded_title)
        return {
            "candidate_normalized_confidence": normalized,
            "candidate_normalized_parse_success": parsed.parse_success,
            "candidate_normalized_options": options if isinstance(options, list) else [],
            "candidate_normalized_prompt_template_id": prompt.prompt_template_id,
            "candidate_normalized_prompt_hash": prompt.prompt_hash,
            "candidate_normalized_raw_output": response.text,
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
        confidence: float,
        grounding: dict[str, Any],
        candidate_normalized: dict[str, Any],
        popularity: dict[str, Any],
        history_similarity: float,
        decision: PolicyDecision,
    ) -> dict[str, Any]:
        grounded_id = grounding.get("grounded_item_id")
        decision_dict = decision.to_dict()
        return {
            "example_id": example.get("example_id"),
            "split": example.get("split"),
            "phase": "phase6_ours_method_integration",
            "ours_method": True,
            "not_paper_result": True,
            "generated_title": generated_title,
            "confidence": confidence,
            "grounding_success": bool(grounding.get("grounding_success")),
            "grounded_item_id": grounded_id,
            "grounding_score": float(grounding.get("grounding_score") or 0.0),
            "grounding_method": grounding.get("grounding_method"),
            "is_catalog_valid": bool(grounding.get("grounding_success")),
            "is_hallucinated": not bool(grounding.get("grounding_success")),
            "is_grounded_hit": bool(grounded_id and grounded_id == str(example["target"])),
            "candidate_adherent": bool(grounded_id and grounded_id in {str(item) for item in example.get("candidates", [])}),
            "uncertainty_decision": decision.decision,
            "uncertainty_reasons": decision_dict["reasons"],
            "uncertainty_risk_flags": decision_dict["risk_flags"],
            "fallback_method": self.fallback.method,
            "echo_risk": bool(decision_dict["risk_flags"].get("echo_risk", False)),
            "popularity_bucket": str(popularity.get("popularity_bucket") or "unknown"),
            "item_popularity": int(popularity.get("item_popularity") or 0),
            "history_similarity": history_similarity,
            "ablation_variant": self.ablation.variant,
            "disabled_components": list(self.ablation.disabled_components),
            "parse_success": parsed_success,
            "parse_error": parse_error,
            **candidate_normalized,
            **prompt_metadata,
            **response_metadata,
        }

    def _accept_prediction(
        self,
        *,
        example: dict[str, Any],
        candidate_items: list[str],
        generated_title: str,
        grounded_id: str | None,
        confidence: float,
        base_metadata: dict[str, Any],
        raw_output: str | None,
    ) -> RankingResult:
        if self.ablation.enabled("grounding_check") and grounded_id:
            predicted_items = [str(grounded_id)]
        elif grounded_id:
            predicted_items = [str(grounded_id)]
        else:
            predicted_items = [generated_title or "__parse_failed__"]
        return RankingResult(
            user_id=str(example["user_id"]),
            target_item=str(example["target"]),
            candidate_items=candidate_items,
            predicted_items=predicted_items,
            scores=[confidence],
            method=self.method_name,
            domain=str(example.get("domain") or "tiny"),
            raw_output=raw_output,
            metadata=base_metadata,
        )

    def _fallback_prediction(
        self,
        *,
        example: dict[str, Any],
        candidate_items: list[str],
        decision: PolicyDecision,
        base_metadata: dict[str, Any],
        raw_output: str | None,
    ) -> RankingResult:
        fallback = self.fallback.rank(example, candidate_items)
        metadata = dict(base_metadata)
        metadata["uncertainty_decision"] = decision.decision
        metadata["fallback_used"] = True
        metadata["fallback_prediction_metadata"] = dict(fallback.metadata)
        return RankingResult(
            user_id=fallback.user_id,
            target_item=fallback.target_item,
            candidate_items=fallback.candidate_items,
            predicted_items=fallback.predicted_items,
            scores=fallback.scores,
            method=self.method_name,
            domain=fallback.domain,
            raw_output=raw_output,
            metadata=metadata,
        )

    def _abstain_prediction(
        self,
        example: dict[str, Any],
        candidate_items: list[str],
        base_metadata: dict[str, Any],
        *,
        raw_output: str | None,
    ) -> RankingResult:
        metadata = dict(base_metadata)
        metadata["fallback_used"] = False
        return RankingResult(
            user_id=str(example["user_id"]),
            target_item=str(example["target"]),
            candidate_items=candidate_items,
            predicted_items=[],
            scores=[],
            method=self.method_name,
            domain=str(example.get("domain") or "tiny"),
            raw_output=raw_output,
            metadata=metadata,
        )

    def _disabled_generation_metadata(self, example: dict[str, Any]) -> dict[str, Any]:
        return {
            "example_id": example.get("example_id"),
            "split": example.get("split"),
            "phase": "phase6_ours_method_integration",
            "ours_method": True,
            "not_paper_result": True,
            "generated_title": "",
            "confidence": 0.0,
            "grounding_success": False,
            "grounded_item_id": None,
            "grounding_score": 0.0,
            "grounding_method": "disabled_fallback_only",
            "is_catalog_valid": False,
            "is_hallucinated": False,
            "is_grounded_hit": False,
            "candidate_adherent": False,
            "uncertainty_decision": "fallback",
            "uncertainty_reasons": ["fallback_only_ablation"],
            "uncertainty_risk_flags": {"echo_risk": False},
            "fallback_method": self.fallback.method,
            "fallback_used": True,
            "echo_risk": False,
            "popularity_bucket": "unknown",
            "item_popularity": 0,
            "history_similarity": 0.0,
            "ablation_variant": self.ablation.variant,
            "disabled_components": list(self.ablation.disabled_components),
            "parse_success": False,
            "parse_error": "generation_disabled_by_fallback_only_ablation",
            "prompt_template_id": "phase6.disabled.fallback_only",
            "prompt_hash": "disabled",
            "candidate_normalized_confidence": None,
            "candidate_normalized_parse_success": None,
            "candidate_normalized_options": [],
            "candidate_normalized_prompt_template_id": None,
            "candidate_normalized_prompt_hash": None,
            "candidate_normalized_raw_output": None,
            "provider": self.provider.provider_name,
            "model": self.provider.model_name,
            "latency_seconds": 0.0,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "cache_hit": False,
            "provider_metadata": {"generation_disabled": True},
        }

    def _popularity_metadata(self, item_id: str | None) -> dict[str, Any]:
        if not item_id:
            return {"item_popularity": 0, "popularity_bucket": "unknown"}
        count = int(self.train_popularity.get(str(item_id), 0))
        positive = sorted({value for value in self.train_popularity.values() if value > 0})
        if not positive or count <= 0:
            bucket = "tail"
        elif count >= positive[-1]:
            bucket = "head"
        elif count <= positive[0]:
            bucket = "tail"
        else:
            bucket = "mid"
        return {"item_popularity": count, "popularity_bucket": bucket}

    def _history_similarity(self, example: dict[str, Any], generated_title: str, grounded_id: str | None) -> float:
        history_titles = []
        for item_id in [str(item) for item in example.get("history", [])]:
            row = self.lookup.get(item_id)
            if row:
                history_titles.append(str(row.get("title") or ""))
        grounded_title = ""
        if grounded_id and grounded_id in self.lookup:
            grounded_title = str(self.lookup[grounded_id].get("title") or "")
        title = grounded_title or generated_title
        return max((token_overlap(title, history_title) for history_title in history_titles), default=0.0)

    def _assert_no_target_leakage(
        self,
        prompt: str,
        example: dict[str, Any],
        prompt_metadata: dict[str, Any] | None = None,
    ) -> None:
        target_id = str(example.get("target") or "")
        metadata = prompt_metadata or {}
        prompt_ids = {
            str(item_id)
            for key in ("history_item_ids", "prompt_candidate_item_ids")
            for item_id in metadata.get(key, [])
        }
        if target_id and target_id in prompt_ids:
            raise ValueError("target item ID leaked into OursMethod prompt")
        target_row = self.lookup.get(target_id)
        target_title = str((target_row or {}).get("title") or "").strip()
        if target_title and target_title.casefold() in prompt.casefold():
            raise ValueError("target title leaked into OursMethod prompt")


def _matched_option_confidence(options: Any, generated_title: str, grounded_title: str | None) -> float | None:
    if not isinstance(options, list):
        return None
    targets = {normalize_title(generated_title)}
    if grounded_title:
        targets.add(normalize_title(grounded_title))
    targets.discard("")
    for option in options:
        if not isinstance(option, dict):
            continue
        title = normalize_title(str(option.get("title") or ""))
        confidence = option.get("confidence")
        if title in targets and isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            return float(confidence)
    return None


def _confidence_or_zero(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    return 0.0


def _train_popularity(train_examples: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for example in train_examples:
        for item_id in [*example.get("history", []), example.get("target")]:
            if item_id:
                counts[str(item_id)] += 1
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

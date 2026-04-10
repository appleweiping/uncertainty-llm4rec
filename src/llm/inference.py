from __future__ import annotations

from typing import Any

from tqdm import tqdm

from src.llm.base import normalize_generation_result
from src.llm.parser import parse_response


def _normalize_candidate_from_pointwise_sample(sample: dict[str, Any]) -> dict[str, Any]:
    candidate_item_id = (
        sample.get("candidate_item_id")
        or sample.get("item_id")
        or sample.get("target_item_id")
        or ""
    )

    candidate_title = (
        sample.get("candidate_title")
        or sample.get("title")
        or ""
    )

    candidate_meta = (
        sample.get("candidate_meta")
        or sample.get("candidate_description")
        or sample.get("candidate_text")
        or sample.get("text")
        or ""
    )

    return {
        "item_id": candidate_item_id,
        "title": candidate_title,
        "meta": candidate_meta,
    }


def _build_result_record(
    *,
    sample: dict[str, Any],
    candidate: dict[str, Any],
    prompt: str,
    raw_text: str,
    generation: dict[str, Any],
    parsed: dict[str, Any],
    label: int,
) -> dict[str, Any]:
    user_id = sample.get("user_id", "")
    target_item_id = (
        sample.get("target_item_id")
        or sample.get("candidate_item_id")
        or candidate.get("item_id", "")
    )
    popularity_group = sample.get("target_popularity_group", "unknown")

    return {
        "user_id": user_id,
        "target_item_id": target_item_id,
        "candidate_item_id": candidate.get("item_id", ""),
        "label": int(label),
        "target_popularity_group": popularity_group,
        "prompt": prompt,
        "raw_response": raw_text,
        "recommend": parsed.get("recommend", "unknown"),
        "confidence": parsed.get("confidence", -1.0),
        "reason": parsed.get("reason", ""),
        "response_latency": generation.get("latency", 0.0),
        "response_model_name": generation.get("model_name", ""),
        "response_provider": generation.get("provider", ""),
        "response_usage": generation.get("usage", {}),
    }


def run_pointwise_inference(
    samples: list[dict],
    llm_backend,
    prompt_builder,
) -> list[dict]:
    results: list[dict] = []

    for sample in tqdm(samples, desc="Running pointwise inference"):
        # 旧 toy schema：sample 里有 target_item + candidates
        if "candidates" in sample:
            user_id = sample["user_id"]
            target_item_id = sample["target_item"]["item_id"]
            popularity_group = sample.get("target_popularity_group", "tail")

            for candidate in sample["candidates"]:
                prompt = prompt_builder.build_pointwise_prompt(sample, candidate)
                generation = normalize_generation_result(
                    llm_backend.generate(prompt),
                    default_provider=getattr(llm_backend, "provider", None),
                    default_model_name=getattr(llm_backend, "model_name", "unknown"),
                )
                raw_text = generation["raw_text"]
                parsed = parse_response(raw_text)

                result = {
                    "user_id": user_id,
                    "target_item_id": target_item_id,
                    "candidate_item_id": candidate["item_id"],
                    "label": int(candidate["item_id"] == target_item_id),
                    "target_popularity_group": popularity_group,
                    "prompt": prompt,
                    "raw_response": raw_text,
                    "recommend": parsed["recommend"],
                    "confidence": parsed["confidence"],
                    "reason": parsed["reason"],
                    "response_latency": generation.get("latency", 0.0),
                    "response_model_name": generation.get("model_name", ""),
                    "response_provider": generation.get("provider", ""),
                    "response_usage": generation.get("usage", {}),
                }
                results.append(result)

            continue

        # 新真实数据 schema：每条 sample 已经是一条 pointwise 记录
        candidate = _normalize_candidate_from_pointwise_sample(sample)
        prompt = prompt_builder.build_pointwise_prompt(sample, candidate)
        generation = normalize_generation_result(
            llm_backend.generate(prompt),
            default_provider=getattr(llm_backend, "provider", None),
            default_model_name=getattr(llm_backend, "model_name", "unknown"),
        )
        raw_text = generation["raw_text"]
        parsed = parse_response(raw_text)

        label = sample.get("label", 0)

        result = _build_result_record(
            sample=sample,
            candidate=candidate,
            prompt=prompt,
            raw_text=raw_text,
            generation=generation,
            parsed=parsed,
            label=label,
        )
        results.append(result)

    return results

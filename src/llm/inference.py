from __future__ import annotations

from typing import Any

from tqdm import tqdm

from src.llm.base import normalize_generation_result
from src.llm.parser import (
    parse_candidate_ranking_response,
    parse_pairwise_preference_response,
    parse_pointwise_response,
)


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


def _generate_with_backend(
    *,
    llm_backend,
    prompt: str,
) -> dict[str, Any]:
    return normalize_generation_result(
        llm_backend.generate(prompt),
        default_provider=getattr(llm_backend, "provider", None),
        default_model_name=getattr(llm_backend, "model_name", "unknown"),
    )


def _attach_generation_fields(
    record: dict[str, Any],
    *,
    prompt: str,
    generation: dict[str, Any],
) -> dict[str, Any]:
    raw_text = generation["raw_text"]
    record.update(
        {
            "prompt": prompt,
            "raw_response": raw_text,
            "latency": generation.get("latency", 0.0),
            "model_name": generation.get("model_name", ""),
            "provider": generation.get("provider", ""),
            "usage": generation.get("usage", {}),
            # Keep response_* for backward compatibility with the old pointwise outputs.
            "response_latency": generation.get("latency", 0.0),
            "response_model_name": generation.get("model_name", ""),
            "response_provider": generation.get("provider", ""),
            "response_usage": generation.get("usage", {}),
        }
    )
    return record


def _build_pointwise_result_record(
    *,
    sample: dict[str, Any],
    candidate: dict[str, Any],
    prompt: str,
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
    parse_success = parsed.get("recommend", "unknown") in {"yes", "no"}

    record = {
        "user_id": user_id,
        "target_item_id": target_item_id,
        "candidate_item_id": candidate.get("item_id", ""),
        "label": int(label),
        "target_popularity_group": popularity_group,
        "recommend": parsed.get("recommend", "unknown"),
        "confidence": parsed.get("confidence", -1.0),
        "reason": parsed.get("reason", ""),
        "parse_success": parse_success,
    }
    return _attach_generation_fields(record, prompt=prompt, generation=generation)


def run_pointwise_inference(
    samples: list[dict[str, Any]],
    llm_backend,
    prompt_builder,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc="Running pointwise inference"):
        if "candidates" in sample:
            user_id = sample["user_id"]
            target_item_id = sample["target_item"]["item_id"]
            popularity_group = sample.get("target_popularity_group", "tail")

            for candidate in sample["candidates"]:
                prompt = prompt_builder.build_pointwise_prompt(sample, candidate)
                generation = _generate_with_backend(llm_backend=llm_backend, prompt=prompt)
                parsed = parse_pointwise_response(generation["raw_text"])

                record = {
                    "user_id": user_id,
                    "target_item_id": target_item_id,
                    "candidate_item_id": candidate["item_id"],
                    "label": int(candidate["item_id"] == target_item_id),
                    "target_popularity_group": popularity_group,
                    "recommend": parsed["recommend"],
                    "confidence": parsed["confidence"],
                    "reason": parsed["reason"],
                    "parse_success": parsed["recommend"] in {"yes", "no"},
                }
                results.append(_attach_generation_fields(record, prompt=prompt, generation=generation))
            continue

        candidate = _normalize_candidate_from_pointwise_sample(sample)
        prompt = prompt_builder.build_pointwise_prompt(sample, candidate)
        generation = _generate_with_backend(llm_backend=llm_backend, prompt=prompt)
        parsed = parse_pointwise_response(generation["raw_text"])

        label = int(sample.get("label", 0))
        results.append(
            _build_pointwise_result_record(
                sample=sample,
                candidate=candidate,
                prompt=prompt,
                generation=generation,
                parsed=parsed,
                label=label,
            )
        )

    return results


def run_candidate_ranking_inference(
    samples: list[dict[str, Any]],
    llm_backend,
    prompt_builder,
    *,
    topk: int | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc="Running candidate ranking inference"):
        candidate_item_ids = [str(item_id) for item_id in sample.get("candidate_item_ids", [])]
        prompt = prompt_builder.build_candidate_ranking_prompt(sample, topk=topk)
        generation = _generate_with_backend(llm_backend=llm_backend, prompt=prompt)
        parsed = parse_candidate_ranking_response(
            generation["raw_text"],
            allowed_item_ids=candidate_item_ids,
            topk=topk,
        )

        record = {
            "user_id": sample.get("user_id", ""),
            "source_event_id": sample.get("source_event_id", ""),
            "positive_item_id": sample.get("positive_item_id", ""),
            "positive_item_index": sample.get("positive_item_index", -1),
            "split_name": sample.get("split_name", ""),
            "timestamp": sample.get("timestamp"),
            "candidate_item_ids": sample.get("candidate_item_ids", []),
            "candidate_titles": sample.get("candidate_titles", []),
            "candidate_popularity_groups": sample.get("candidate_popularity_groups", []),
            "pred_ranked_item_ids": parsed.get("ranked_item_ids", []),
            "topk_item_ids": parsed.get("topk_item_ids", []),
            "confidence": parsed.get("confidence", -1.0),
            "reason": parsed.get("reason", ""),
            "parse_mode": parsed.get("parse_mode", "failed"),
            "parse_success": bool(parsed.get("parse_success", False)),
            "contains_out_of_candidate_item": bool(parsed.get("contains_out_of_candidate_item", False)),
            "out_of_candidate_item_ids": parsed.get("out_of_candidate_item_ids", []),
        }
        results.append(_attach_generation_fields(record, prompt=prompt, generation=generation))

    return results


def run_pairwise_preference_inference(
    samples: list[dict[str, Any]],
    llm_backend,
    prompt_builder,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc="Running pairwise preference inference"):
        prompt = prompt_builder.build_pairwise_preference_prompt(sample)
        generation = _generate_with_backend(llm_backend=llm_backend, prompt=prompt)
        parsed = parse_pairwise_preference_response(
            generation["raw_text"],
            item_a_id=str(sample.get("item_a_id", "")),
            item_b_id=str(sample.get("item_b_id", "")),
        )

        record = {
            "pair_id": sample.get("pair_id", ""),
            "source_event_id": sample.get("source_event_id", ""),
            "user_id": sample.get("user_id", ""),
            "item_a_id": sample.get("item_a_id", ""),
            "item_b_id": sample.get("item_b_id", ""),
            "preferred_item_true": sample.get("preferred_item", ""),
            "preferred_item_pred": parsed.get("preferred_item", ""),
            "confidence": parsed.get("confidence", -1.0),
            "reason": parsed.get("reason", ""),
            "pair_type": sample.get("pair_type", ""),
            "split_name": sample.get("split_name", ""),
            "timestamp": sample.get("timestamp"),
            "parse_mode": parsed.get("parse_mode", "failed"),
            "parse_success": bool(parsed.get("parse_success", False)),
            "ambiguous_preference": bool(parsed.get("ambiguous_preference", False)),
        }
        results.append(_attach_generation_fields(record, prompt=prompt, generation=generation))

    return results


def run_task_inference(
    *,
    samples: list[dict[str, Any]],
    llm_backend,
    prompt_builder,
    task_type: str,
    topk: int | None = None,
) -> list[dict[str, Any]]:
    normalized_task_type = str(task_type).strip().lower()

    if normalized_task_type in {"pointwise", "pointwise_yesno"}:
        return run_pointwise_inference(samples=samples, llm_backend=llm_backend, prompt_builder=prompt_builder)
    if normalized_task_type == "candidate_ranking":
        return run_candidate_ranking_inference(
            samples=samples,
            llm_backend=llm_backend,
            prompt_builder=prompt_builder,
            topk=topk,
        )
    if normalized_task_type == "pairwise_preference":
        return run_pairwise_preference_inference(
            samples=samples,
            llm_backend=llm_backend,
            prompt_builder=prompt_builder,
        )

    raise ValueError(f"Unsupported task_type for inference: {task_type}")

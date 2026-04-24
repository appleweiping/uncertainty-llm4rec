from __future__ import annotations

from typing import Any

from tqdm import tqdm

from src.llm.base import normalize_generation_result
from src.llm.parser import (
    parse_candidate_ranking_response,
    parse_pairwise_preference_response,
    parse_pointwise_response,
)
from src.shadow import compute_shadow_scores, parse_shadow_response
from src.utils.io import save_jsonl


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


def _batch_generate_with_backend(
    *,
    llm_backend,
    prompts: list[str],
    **kwargs,
) -> list[dict[str, Any]]:
    raw_results = llm_backend.batch_generate(prompts, **kwargs)
    return [
        normalize_generation_result(
            result,
            default_provider=getattr(llm_backend, "provider", None),
            default_model_name=getattr(llm_backend, "model_name", "unknown"),
        )
        for result in raw_results
    ]


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


def _build_shadow_pointwise_result_record(
    *,
    sample: dict[str, Any],
    candidate: dict[str, Any],
    prompt: str,
    generation: dict[str, Any],
    parsed: dict[str, Any],
    label: int,
    shadow_variant: str,
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    user_id = sample.get("user_id", "")
    target_item_id = (
        sample.get("target_item_id")
        or sample.get("candidate_item_id")
        or candidate.get("item_id", "")
    )
    popularity_group = sample.get("target_popularity_group", "unknown")
    score_fields = compute_shadow_scores(parsed, variant=shadow_variant)
    shadow_score = float(score_fields.get("shadow_score", parsed.get("shadow_primary_score", 0.0)))
    recommend = "yes" if shadow_score >= float(decision_threshold) else "no"

    record = {
        "user_id": user_id,
        "target_item_id": target_item_id,
        "candidate_item_id": candidate.get("item_id", ""),
        "label": int(label),
        "target_popularity_group": popularity_group,
        "recommend": recommend,
        # Keep confidence as the task-grounded score for legacy table compatibility.
        "confidence": shadow_score,
        "reason": parsed.get("reason", ""),
        "parse_success": bool(parsed.get("parse_success", False)),
        **parsed,
        **score_fields,
    }
    return _attach_generation_fields(record, prompt=prompt, generation=generation)


def run_pointwise_inference(
    samples: list[dict[str, Any]],
    llm_backend,
    prompt_builder,
    *,
    checkpoint_path=None,
    checkpoint_every_batches: int = 1,
    existing_records: list[dict[str, Any]] | None = None,
    response_schema: str = "pointwise_yesno",
    shadow_variant: str | None = None,
    decision_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = list(existing_records or [])
    batch_size = int(getattr(llm_backend, "batch_size", 1) or 1)
    checkpoint_every_batches = max(1, int(checkpoint_every_batches))

    pointwise_jobs: list[dict[str, Any]] = []
    for sample in samples:
        if "candidates" in sample:
            user_id = sample["user_id"]
            target_item_id = sample["target_item"]["item_id"]
            popularity_group = sample.get("target_popularity_group", "tail")

            for candidate in sample["candidates"]:
                pointwise_jobs.append(
                    {
                        "sample": sample,
                        "candidate": candidate,
                        "label": int(candidate["item_id"] == target_item_id),
                        "user_id": user_id,
                        "target_item_id": target_item_id,
                        "target_popularity_group": popularity_group,
                        "legacy_candidate_mode": True,
                    }
                )
            continue

        pointwise_jobs.append(
            {
                "sample": sample,
                "candidate": _normalize_candidate_from_pointwise_sample(sample),
                "label": int(sample.get("label", 0)),
                "legacy_candidate_mode": False,
            }
        )

    if existing_records:
        resume_count = min(len(existing_records), len(pointwise_jobs))
        pointwise_jobs = pointwise_jobs[resume_count:]

    progress = tqdm(total=len(pointwise_jobs), desc="Running pointwise inference")
    try:
        batch_counter = 0
        for start_idx in range(0, len(pointwise_jobs), batch_size):
            batch_jobs = pointwise_jobs[start_idx : start_idx + batch_size]
            batch_prompts = [
                prompt_builder.build_pointwise_prompt(job["sample"], job["candidate"])
                for job in batch_jobs
            ]
            batch_generations = _batch_generate_with_backend(
                llm_backend=llm_backend,
                prompts=batch_prompts,
            )

            for job, prompt, generation in zip(batch_jobs, batch_prompts, batch_generations):
                if response_schema == "shadow":
                    if not shadow_variant:
                        raise ValueError("shadow_variant must be provided when response_schema=shadow.")
                    parsed = parse_shadow_response(generation["raw_text"], variant=shadow_variant)
                    if job["legacy_candidate_mode"]:
                        record = {
                            "user_id": job["user_id"],
                            "target_item_id": job["target_item_id"],
                            "candidate_item_id": job["candidate"]["item_id"],
                            "label": job["label"],
                            "target_popularity_group": job["target_popularity_group"],
                        }
                        shadow_record = _build_shadow_pointwise_result_record(
                            sample=record,
                            candidate=job["candidate"],
                            prompt=prompt,
                            generation=generation,
                            parsed=parsed,
                            label=job["label"],
                            shadow_variant=shadow_variant,
                            decision_threshold=decision_threshold,
                        )
                        results.append(shadow_record)
                        continue

                    results.append(
                        _build_shadow_pointwise_result_record(
                            sample=job["sample"],
                            candidate=job["candidate"],
                            prompt=prompt,
                            generation=generation,
                            parsed=parsed,
                            label=job["label"],
                            shadow_variant=shadow_variant,
                            decision_threshold=decision_threshold,
                        )
                    )
                    continue

                parsed = parse_pointwise_response(generation["raw_text"])
                if job["legacy_candidate_mode"]:
                    record = {
                        "user_id": job["user_id"],
                        "target_item_id": job["target_item_id"],
                        "candidate_item_id": job["candidate"]["item_id"],
                        "label": job["label"],
                        "target_popularity_group": job["target_popularity_group"],
                        "recommend": parsed["recommend"],
                        "confidence": parsed["confidence"],
                        "reason": parsed["reason"],
                        "parse_success": parsed["recommend"] in {"yes", "no"},
                    }
                    results.append(
                        _attach_generation_fields(record, prompt=prompt, generation=generation)
                    )
                    continue

                results.append(
                    _build_pointwise_result_record(
                        sample=job["sample"],
                        candidate=job["candidate"],
                        prompt=prompt,
                        generation=generation,
                        parsed=parsed,
                        label=job["label"],
                    )
                )
            progress.update(len(batch_jobs))
            batch_counter += 1
            if checkpoint_path is not None and batch_counter % checkpoint_every_batches == 0:
                save_jsonl(results, checkpoint_path)
    finally:
        progress.close()

    return results


def run_candidate_ranking_inference(
    samples: list[dict[str, Any]],
    llm_backend,
    prompt_builder,
    *,
    topk: int | None = None,
    max_new_tokens: int | None = None,
    checkpoint_path=None,
    checkpoint_every_batches: int = 1,
    existing_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = list(existing_records or [])
    batch_size = int(getattr(llm_backend, "batch_size", 1) or 1)
    checkpoint_every_batches = max(1, int(checkpoint_every_batches))

    progress = tqdm(total=len(samples), desc="Running candidate ranking inference")
    try:
        batch_counter = 0
        for start_idx in range(0, len(samples), batch_size):
            batch_samples = samples[start_idx : start_idx + batch_size]
            batch_prompts = [
                prompt_builder.build_candidate_ranking_prompt(sample, topk=topk)
                for sample in batch_samples
            ]
            batch_generations = _batch_generate_with_backend(
                llm_backend=llm_backend,
                prompts=batch_prompts,
                max_new_tokens=max_new_tokens,
            )

            for sample, prompt, generation in zip(batch_samples, batch_prompts, batch_generations):
                candidate_item_ids = [str(item_id) for item_id in sample.get("candidate_item_ids", [])]
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
            progress.update(len(batch_samples))
            batch_counter += 1
            if checkpoint_path is not None and batch_counter % checkpoint_every_batches == 0:
                save_jsonl(results, checkpoint_path)
    finally:
        progress.close()

    return results


def run_pairwise_preference_inference(
    samples: list[dict[str, Any]],
    llm_backend,
    prompt_builder,
    *,
    checkpoint_path=None,
    checkpoint_every_batches: int = 1,
    existing_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = list(existing_records or [])
    batch_size = int(getattr(llm_backend, "batch_size", 1) or 1)
    checkpoint_every_batches = max(1, int(checkpoint_every_batches))

    progress = tqdm(total=len(samples), desc="Running pairwise preference inference")
    try:
        batch_counter = 0
        for start_idx in range(0, len(samples), batch_size):
            batch_samples = samples[start_idx : start_idx + batch_size]
            batch_prompts = [
                prompt_builder.build_pairwise_preference_prompt(sample)
                for sample in batch_samples
            ]
            batch_generations = _batch_generate_with_backend(
                llm_backend=llm_backend,
                prompts=batch_prompts,
            )

            for sample, prompt, generation in zip(batch_samples, batch_prompts, batch_generations):
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
                results.append(
                    _attach_generation_fields(record, prompt=prompt, generation=generation)
                )
            progress.update(len(batch_samples))
            batch_counter += 1
            if checkpoint_path is not None and batch_counter % checkpoint_every_batches == 0:
                save_jsonl(results, checkpoint_path)
    finally:
        progress.close()

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

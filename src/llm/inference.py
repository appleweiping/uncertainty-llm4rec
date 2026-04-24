from __future__ import annotations

from typing import Any

from tqdm import tqdm

from src.llm.base import normalize_generation_result
from src.llm.parser import parse_evidence_response, parse_response


EVIDENCE_OUTPUT_SCHEMAS = {
    "evidence",
    "evidence_confidence",
    "evidence_posterior",
    "calibrated_evidence_posterior",
}

EVIDENCE_RESULT_KEYS = (
    "positive_evidence",
    "negative_evidence",
    "evidence_margin",
    "abs_evidence_margin",
    "ambiguity",
    "missing_information",
    "raw_confidence",
    "parse_success",
    "parse_error",
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


def _parse_pointwise_response(raw_text: str, output_schema: str | None) -> dict[str, Any]:
    schema = str(output_schema or "verbalized_confidence").strip().lower()
    if schema in EVIDENCE_OUTPUT_SCHEMAS:
        return parse_evidence_response(raw_text)
    return parse_response(raw_text)


def _add_optional_evidence_fields(result: dict[str, Any], parsed: dict[str, Any]) -> None:
    for key in EVIDENCE_RESULT_KEYS:
        if key in parsed:
            result[key] = parsed[key]


def _build_result_record(
    *,
    sample: dict[str, Any],
    candidate: dict[str, Any],
    prompt: str,
    raw_text: str,
    generation: dict[str, Any],
    parsed: dict[str, Any],
    label: int,
    method_variant: str | None = None,
) -> dict[str, Any]:
    user_id = sample.get("user_id", "")
    target_item_id = (
        sample.get("target_item_id")
        or sample.get("candidate_item_id")
        or candidate.get("item_id", "")
    )
    popularity_group = sample.get("target_popularity_group", "unknown")

    result = {
        "user_id": user_id,
        "target_item_id": target_item_id,
        "candidate_item_id": candidate.get("item_id", ""),
        "label": int(label),
        "target_popularity_group": popularity_group,
        "prompt": prompt,
        "raw_response": raw_text,
        "recommend": parsed.get("recommend", "unknown"),
        "confidence": parsed.get("confidence", parsed.get("raw_confidence", -1.0)),
        "reason": parsed.get("reason", ""),
        "response_latency": generation.get("latency", 0.0),
        "response_model_name": generation.get("model_name", ""),
        "response_provider": generation.get("provider", ""),
        "response_usage": generation.get("usage", {}),
    }
    if method_variant:
        result["method_variant"] = method_variant
    _add_optional_evidence_fields(result, parsed)
    return result


def run_pointwise_inference(
    samples: list[dict],
    llm_backend,
    prompt_builder,
    output_schema: str | None = None,
    method_variant: str | None = None,
) -> list[dict]:
    results: list[dict] = []

    for sample in tqdm(samples, desc="Running pointwise inference"):
        # Legacy toy schema: one record contains target_item and multiple candidates.
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
                parsed = _parse_pointwise_response(raw_text, output_schema)

                result = {
                    "user_id": user_id,
                    "target_item_id": target_item_id,
                    "candidate_item_id": candidate["item_id"],
                    "label": int(candidate["item_id"] == target_item_id),
                    "target_popularity_group": popularity_group,
                    "prompt": prompt,
                    "raw_response": raw_text,
                    "recommend": parsed.get("recommend", "unknown"),
                    "confidence": parsed.get("confidence", parsed.get("raw_confidence", -1.0)),
                    "reason": parsed.get("reason", ""),
                    "response_latency": generation.get("latency", 0.0),
                    "response_model_name": generation.get("model_name", ""),
                    "response_provider": generation.get("provider", ""),
                    "response_usage": generation.get("usage", {}),
                }
                if method_variant:
                    result["method_variant"] = method_variant
                _add_optional_evidence_fields(result, parsed)
                results.append(result)

            continue

        # Current pointwise schema: each record is already a user-candidate sample.
        candidate = _normalize_candidate_from_pointwise_sample(sample)
        prompt = prompt_builder.build_pointwise_prompt(sample, candidate)
        generation = normalize_generation_result(
            llm_backend.generate(prompt),
            default_provider=getattr(llm_backend, "provider", None),
            default_model_name=getattr(llm_backend, "model_name", "unknown"),
        )
        raw_text = generation["raw_text"]
        parsed = _parse_pointwise_response(raw_text, output_schema)

        label = sample.get("label", 0)

        result = _build_result_record(
            sample=sample,
            candidate=candidate,
            prompt=prompt,
            raw_text=raw_text,
            generation=generation,
            parsed=parsed,
            label=label,
            method_variant=method_variant,
        )
        results.append(result)

    return results

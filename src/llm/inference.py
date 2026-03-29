# src/llm/inference.py
from __future__ import annotations

from typing import Any

from tqdm import tqdm

from src.llm.parser import parse_response


def run_pointwise_inference(
    samples: list[dict[str, Any]],
    llm_backend,
    prompt_builder,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc="Running pointwise inference"):
        user_id = sample["user_id"]
        target_item_id = sample["target_item"]["item_id"]
        popularity_group = sample.get("target_popularity_group", "tail")

        for candidate in sample["candidates"]:
            prompt = prompt_builder.build_pointwise_prompt(sample, candidate)
            raw_text = llm_backend.generate(prompt)
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
            }
            results.append(result)

    return results
# src/llm/self_consistency.py

from __future__ import annotations

from typing import List, Dict, Any
import time

from src.llm.base import normalize_generation_result
from src.llm.parser import parse_response


def run_self_consistency(
    llm_backend,
    prompt: str,
    num_samples: int = 5,
    sleep_time: float = 0.0,
    generation_kwargs: dict[str, Any] | None = None,
) -> List[Dict]:
    """
    Run multiple LLM generations for the same prompt and return
    parsed structured outputs for consistency analysis.
    """
    outputs = []
    generation_kwargs = generation_kwargs or {}

    for _ in range(num_samples):
        generation = normalize_generation_result(
            llm_backend.generate(prompt, **generation_kwargs),
            default_provider=getattr(llm_backend, "provider", None),
            default_model_name=getattr(llm_backend, "model_name", "unknown"),
        )
        raw_text = generation["raw_text"]
        parsed = parse_response(raw_text)

        outputs.append(
            {
                "raw_response": raw_text,
                "recommend": parsed.get("recommend", "unknown"),
                "confidence": parsed.get("confidence", -1.0),
                "reason": parsed.get("reason", ""),
                "response_latency": generation.get("latency", 0.0),
                "response_model_name": generation.get("model_name", ""),
                "response_provider": generation.get("provider", ""),
                "response_usage": generation.get("usage", {}),
            }
        )

        if sleep_time > 0:
            time.sleep(sleep_time)

    return outputs

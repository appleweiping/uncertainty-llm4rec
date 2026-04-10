# src/llm/self_consistency.py

from __future__ import annotations

from typing import List, Dict
import time

from src.llm.base import normalize_generation_result


def run_self_consistency(
    llm_backend,
    prompt: str,
    num_samples: int = 5,
    sleep_time: float = 0.0
) -> List[Dict]:
    """
    Run multiple LLM generations for the same prompt.
    """
    outputs = []

    for _ in range(num_samples):
        response = normalize_generation_result(
            llm_backend.generate(prompt),
            default_provider=getattr(llm_backend, "provider", None),
            default_model_name=getattr(llm_backend, "model_name", "unknown"),
        )
        outputs.append(response)

        if sleep_time > 0:
            time.sleep(sleep_time)

    return outputs

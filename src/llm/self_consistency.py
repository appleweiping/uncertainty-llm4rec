# src/llm/self_consistency.py

from __future__ import annotations

from collections.abc import Callable
from typing import List, Dict, Any
import time

from src.llm.base import normalize_generation_result
from src.llm.parser import parse_evidence_response, parse_response


def _parse_self_consistency_response(raw_text: str, output_schema: str | None = None) -> dict[str, Any]:
    schema = str(output_schema or "").strip().lower()
    if schema == "evidence_confidence":
        parsed = parse_evidence_response(raw_text)
        return {
            "recommend": parsed.get("recommend", "unknown"),
            "confidence": parsed.get("raw_confidence", -1.0),
            "raw_confidence": parsed.get("raw_confidence"),
            "positive_evidence": parsed.get("positive_evidence"),
            "negative_evidence": parsed.get("negative_evidence"),
            "evidence_margin": parsed.get("evidence_margin"),
            "abs_evidence_margin": parsed.get("abs_evidence_margin"),
            "ambiguity": parsed.get("ambiguity"),
            "missing_information": parsed.get("missing_information"),
            "reason": parsed.get("reason", ""),
            "parse_success": bool(parsed.get("parse_success", False)),
            "parse_error": parsed.get("parse_error", ""),
        }

    parsed = parse_response(raw_text)
    return {
        "recommend": parsed.get("recommend", "unknown"),
        "confidence": parsed.get("confidence", -1.0),
        "reason": parsed.get("reason", ""),
        "parse_success": parsed.get("recommend", "unknown") in {"yes", "no"},
        "parse_error": "" if parsed.get("recommend", "unknown") in {"yes", "no"} else "invalid_recommend",
    }


def run_self_consistency(
    llm_backend,
    prompt: str,
    num_samples: int = 5,
    sleep_time: float = 0.0,
    generation_kwargs: dict[str, Any] | None = None,
    output_schema: str | None = None,
    max_retries: int = 0,
    retry_backoff_seconds: float = 2.0,
    before_generate: Callable[[], None] | None = None,
) -> List[Dict]:
    """
    Run multiple LLM generations for the same prompt and return
    parsed structured outputs for consistency analysis.
    """
    outputs = []
    generation_kwargs = generation_kwargs or {}

    for trial_index in range(num_samples):
        output: dict[str, Any] | None = None
        for attempt in range(max(0, int(max_retries)) + 1):
            try:
                if before_generate is not None:
                    before_generate()
                generation = normalize_generation_result(
                    llm_backend.generate(prompt, **generation_kwargs),
                    default_provider=getattr(llm_backend, "provider", None),
                    default_model_name=getattr(llm_backend, "model_name", "unknown"),
                )
                raw_text = generation["raw_text"]
                parsed = _parse_self_consistency_response(raw_text, output_schema=output_schema)
                output = {
                    "trial_index": int(trial_index),
                    "retry_count": int(attempt),
                    "raw_response": raw_text,
                    **parsed,
                    "response_latency": generation.get("latency", 0.0),
                    "response_model_name": generation.get("model_name", ""),
                    "response_provider": generation.get("provider", ""),
                    "response_usage": generation.get("usage", {}),
                }
                if output.get("parse_success", False) or attempt >= max_retries:
                    break
            except Exception as exc:
                output = {
                    "trial_index": int(trial_index),
                    "retry_count": int(attempt),
                    "raw_response": "",
                    "recommend": "unknown",
                    "confidence": -1.0,
                    "reason": "",
                    "parse_success": False,
                    "parse_error": type(exc).__name__,
                    "response_latency": 0.0,
                    "response_model_name": getattr(llm_backend, "model_name", ""),
                    "response_provider": getattr(llm_backend, "provider", ""),
                    "response_usage": {},
                }
                if attempt >= max_retries:
                    break

            if attempt < max_retries:
                time.sleep(retry_backoff_seconds * (2 ** attempt))

        outputs.append(output or {"trial_index": int(trial_index), "recommend": "unknown", "confidence": -1.0})

        if sleep_time > 0:
            time.sleep(sleep_time)

    return outputs

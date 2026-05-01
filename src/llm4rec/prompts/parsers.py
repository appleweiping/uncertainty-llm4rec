"""Robust parsers for structured LLM responses."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ParseResult:
    data: dict[str, Any]
    parse_success: bool
    raw_output: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_llm_json(text: str, *, allow_confidence_clamp: bool = False) -> ParseResult:
    raw = text or ""
    try:
        payload = _extract_json_object(raw)
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("parsed JSON must be an object")
        _validate_confidences(data, allow_clamp=allow_confidence_clamp)
        return ParseResult(data=data, parse_success=True, raw_output=raw)
    except Exception as exc:  # noqa: BLE001 - parser failures must stay non-fatal.
        return ParseResult(data={}, parse_success=False, raw_output=raw, error=str(exc))


def parse_generation_response(text: str, *, allow_confidence_clamp: bool = False) -> ParseResult:
    result = parse_llm_json(text, allow_confidence_clamp=allow_confidence_clamp)
    if not result.parse_success:
        return result
    recommendation = result.data.get("recommendation")
    if not isinstance(recommendation, str):
        return _failed(result, "recommendation must be a string")
    if not recommendation.strip():
        return _failed(result, "recommendation must be non-empty")
    if "confidence" not in result.data:
        return _failed(result, "confidence is required")
    return result


def parse_rerank_response(text: str, *, allow_confidence_clamp: bool = False) -> ParseResult:
    result = parse_llm_json(text, allow_confidence_clamp=allow_confidence_clamp)
    if not result.parse_success:
        return result
    ranked = result.data.get("ranked_items")
    if not isinstance(ranked, list):
        return _failed(result, "ranked_items must be a list")
    if not ranked:
        return _failed(result, "ranked_items must be non-empty")
    for index, row in enumerate(ranked):
        if not isinstance(row, dict) or not isinstance(row.get("title"), str):
            return _failed(result, f"ranked_items[{index}].title must be a string")
    return result


def parse_yes_no_response(text: str, *, allow_confidence_clamp: bool = False) -> ParseResult:
    result = parse_llm_json(text, allow_confidence_clamp=allow_confidence_clamp)
    if not result.parse_success:
        return result
    answer = str(result.data.get("answer") or "").lower()
    if answer not in {"yes", "no"}:
        return _failed(result, "answer must be yes or no")
    if "confidence" not in result.data:
        return _failed(result, "confidence is required")
    return result


def parse_candidate_normalized_response(text: str, *, allow_confidence_clamp: bool = False) -> ParseResult:
    result = parse_llm_json(text, allow_confidence_clamp=allow_confidence_clamp)
    if not result.parse_success:
        return result
    options = result.data.get("options")
    if not isinstance(options, list):
        return _failed(result, "options must be a list")
    if not options:
        return _failed(result, "options must be non-empty")
    for index, row in enumerate(options):
        if not isinstance(row, dict) or not isinstance(row.get("title"), str):
            return _failed(result, f"options[{index}].title must be a string")
    return result


def _extract_json_object(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise ValueError("unterminated JSON object")


def _validate_confidences(value: Any, *, allow_clamp: bool) -> Any:
    if isinstance(value, dict):
        for key, child in list(value.items()):
            if key == "confidence":
                if isinstance(child, bool) or not isinstance(child, (int, float)):
                    raise ValueError("confidence must be numeric")
                confidence = float(child)
                if not math.isfinite(confidence):
                    raise ValueError("confidence must be finite")
                if not 0.0 <= confidence <= 1.0:
                    if allow_clamp:
                        value[key] = min(1.0, max(0.0, confidence))
                    else:
                        raise ValueError("confidence must be in [0, 1]")
            else:
                _validate_confidences(child, allow_clamp=allow_clamp)
    elif isinstance(value, list):
        for child in value:
            _validate_confidences(child, allow_clamp=allow_clamp)
    return value


def _failed(result: ParseResult, error: str) -> ParseResult:
    return ParseResult(
        data=result.data,
        parse_success=False,
        raw_output=result.raw_output,
        error=error,
        metadata=result.metadata,
    )

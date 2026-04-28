"""Robust parsing for generated title and confidence API responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_TITLE_RE = re.compile(
    r"(?:generated_title|title|recommendation|recommended title)\s*[:=]\s*[\"']?(.+?)[\"']?(?:\n|,|$)",
    re.IGNORECASE,
)
_CONF_RE = re.compile(
    r"(?:confidence|probability(?:\s+correct)?|probability)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?\s*%?)",
    re.IGNORECASE,
)
_YES_NO_RE = re.compile(
    r"(?:is_likely_correct|likely_correct|correct|answer|yes/no)\s*[:=]\s*([A-Za-z]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ParsedObservationResponse:
    """Parsed title, confidence, and yes/no self-verification."""

    success: bool
    raw_text: str
    parse_strategy: str
    generated_title: str | None = None
    confidence: float | None = None
    is_likely_correct: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def normalize_confidence(value: Any) -> float:
    """Normalize confidence in 0-1, 0-100, or percentage string formats."""

    if value is None:
        raise ValueError("confidence is missing")
    if isinstance(value, str):
        text = value.strip()
        is_percent = text.endswith("%")
        text = text.rstrip("%").strip()
        number = float(text)
        if is_percent or number > 1.0:
            number = number / 100.0
    else:
        number = float(value)
        if number > 1.0:
            number = number / 100.0
    if not 0.0 <= number <= 1.0:
        raise ValueError("confidence must normalize to [0, 1]")
    return number


def normalize_yes_no(value: Any) -> str:
    """Normalize common yes/no variants."""

    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value).strip().lower()
    yes_values = {"yes", "y", "true", "correct", "likely", "likely_correct", "1"}
    no_values = {"no", "n", "false", "incorrect", "unlikely", "not_likely", "0"}
    if text in yes_values:
        return "yes"
    if text in no_values:
        return "no"
    raise ValueError(f"unsupported yes/no value: {value}")


def _extract_payload_fields(payload: dict[str, Any], *, strategy: str, raw_text: str) -> ParsedObservationResponse:
    title = (
        payload.get("generated_title")
        or payload.get("title")
        or payload.get("recommendation")
        or payload.get("recommended_title")
    )
    generated_title = str(title or "").strip()
    if not generated_title:
        raise ValueError("generated title is empty or missing")
    confidence = normalize_confidence(
        payload.get("confidence")
        if "confidence" in payload
        else payload.get("probability_correct", payload.get("probability"))
    )
    yes_no_source = (
        payload.get("is_likely_correct")
        if "is_likely_correct" in payload
        else payload.get("likely_correct", payload.get("correct", "yes"))
    )
    yes_no = normalize_yes_no(yes_no_source)
    return ParsedObservationResponse(
        success=True,
        raw_text=raw_text,
        parse_strategy=strategy,
        generated_title=generated_title,
        confidence=confidence,
        is_likely_correct=yes_no,
        payload=dict(payload),
    )


def _try_json(raw_text: str, *, strategy: str) -> ParsedObservationResponse | None:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return _extract_payload_fields(payload, strategy=strategy, raw_text=raw_text)


def _try_fenced_json(raw_text: str) -> ParsedObservationResponse | None:
    match = _FENCED_JSON_RE.search(raw_text)
    if not match:
        return None
    return _try_json(match.group(1), strategy="fenced_json")


def _try_embedded_json(raw_text: str) -> ParsedObservationResponse | None:
    match = _OBJECT_RE.search(raw_text)
    if not match:
        return None
    return _try_json(match.group(0), strategy="embedded_json")


def _try_regex(raw_text: str) -> ParsedObservationResponse | None:
    title_match = _TITLE_RE.search(raw_text)
    confidence_match = _CONF_RE.search(raw_text)
    yes_no_match = _YES_NO_RE.search(raw_text)
    if not title_match or not confidence_match:
        return None
    payload = {
        "generated_title": title_match.group(1).strip().strip('"').strip("'"),
        "confidence": confidence_match.group(1).strip(),
        "is_likely_correct": yes_no_match.group(1) if yes_no_match else "yes",
    }
    return _extract_payload_fields(payload, strategy="regex", raw_text=raw_text)


def parse_observation_response(raw_text: str) -> ParsedObservationResponse:
    """Parse strict JSON, fenced JSON, embedded JSON, then regex fallback."""

    text = str(raw_text or "").strip()
    if not text:
        return ParsedObservationResponse(
            success=False,
            raw_text=raw_text,
            parse_strategy="none",
            error="empty response",
        )
    strategies = (
        lambda: _try_json(text, strategy="strict_json"),
        lambda: _try_fenced_json(text),
        lambda: _try_embedded_json(text),
        lambda: _try_regex(text),
    )
    last_error: str | None = None
    for strategy in strategies:
        try:
            parsed = strategy()
        except Exception as exc:  # noqa: BLE001 - parser must capture failures.
            last_error = str(exc)
            continue
        if parsed is not None:
            return parsed
    return ParsedObservationResponse(
        success=False,
        raw_text=raw_text,
        parse_strategy="failed",
        error=last_error or "could not parse generated title/confidence",
    )

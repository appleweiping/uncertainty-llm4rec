from __future__ import annotations

import json
import re
from typing import Any

from src.llm.parser import strip_thinking_blocks
from src.shadow.schema import get_shadow_variant


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _extract_json_payload(clean: str) -> dict[str, Any] | None:
    try:
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start != -1 and end > start:
            payload = json.loads(clean[start:end])
            return payload if isinstance(payload, dict) else None
    except Exception:
        return None
    return None


def _clamp01(value: Any, default: float = -1.0) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, str):
            text = value.strip().replace("%", "")
            if text == "":
                return default
            parsed = float(text)
        else:
            parsed = float(value)
    except Exception:
        return default
    if parsed > 1.0:
        parsed = parsed / 100.0
    return max(0.0, min(1.0, parsed))


def _clamp_signed(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(str(value).strip())
    except Exception:
        return default
    return max(-1.0, min(1.0, parsed))


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    return normalized in {"true", "1", "yes", "y", "fallback", "anchor"}


def parse_shadow_response(text: str, *, variant: str) -> dict[str, Any]:
    spec = get_shadow_variant(variant)
    raw = strip_thinking_blocks(text)
    clean = _strip_code_fence(raw)
    obj = _extract_json_payload(clean) or {}

    parsed: dict[str, Any] = {
        "shadow_variant": spec.variant,
        "shadow_method_name": spec.method_name,
        "parse_success": bool(obj),
        "parse_mode": "json" if obj else "failed",
        "reason": str(obj.get("reason", "")).strip() if obj else "",
    }

    for field in spec.output_fields:
        if field == "reason":
            continue
        if field in {"cutoff_margin_estimate"}:
            parsed[field] = _clamp_signed(obj.get(field), default=0.0)
        elif field in {"fallback_flag"}:
            parsed[field] = _normalize_bool(obj.get(field))
        elif field in {"intent_prototype", "pair_type"}:
            parsed[field] = str(obj.get(field, "")).strip()
        else:
            parsed[field] = _clamp01(obj.get(field), default=-1.0)

    primary_value = parsed.get(spec.primary_score_field, -1.0)
    parsed["shadow_primary_score"] = primary_value
    if primary_value == -1.0:
        parsed["parse_success"] = False
    return parsed

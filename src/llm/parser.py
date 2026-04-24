from __future__ import annotations

import json
import re
from typing import Any


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _normalize_confidence(value: Any) -> float:
    if value is None:
        return -1.0

    if isinstance(value, (int, float)):
        conf = float(value)
    else:
        text = str(value).strip().replace("%", "")
        if text == "":
            return -1.0
        conf = float(text)

    if conf > 1.0:
        conf = conf / 100.0

    conf = max(0.0, min(1.0, conf))
    return conf


def _normalize_optional_unit_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        if isinstance(value, (int, float)):
            number = float(value)
        else:
            text = str(value).strip().replace("%", "")
            if text == "":
                return None
            number = float(text)
    except (TypeError, ValueError):
        return None

    if number > 1.0:
        number = number / 100.0

    return max(0.0, min(1.0, number))


def _extract_json_object(clean: str) -> dict[str, Any] | None:
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start == -1 or end <= start:
        return None

    try:
        parsed = json.loads(clean[start:end])
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _regex_field(clean: str, field_name: str) -> str | None:
    pattern = rf'"?{re.escape(field_name)}"?\s*[:=]\s*"?(.*?)"?\s*(?:,|\n|$)'
    match = re.search(pattern, clean, re.I)
    return match.group(1).strip() if match else None


def _regex_number_field(clean: str, field_name: str) -> str | None:
    pattern = rf'"?{re.escape(field_name)}"?\s*[:=]\s*"?([0-9]*\.?[0-9]+%?)"?'
    match = re.search(pattern, clean, re.I)
    return match.group(1) if match else None


def _normalize_recommend(value: Any) -> str:
    recommend = str(value or "unknown").strip().lower()
    return recommend if recommend in {"yes", "no"} else "unknown"


def parse_response(text: str) -> dict[str, Any]:
    raw = text.strip()
    clean = _strip_code_fence(raw)

    try:
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start != -1 and end > start:
            obj = json.loads(clean[start:end])
            recommend = str(obj.get("recommend", "unknown")).strip().lower()
            confidence = _normalize_confidence(obj.get("confidence", -1.0))
            reason = str(obj.get("reason", "")).strip()
            return {
                "recommend": recommend if recommend in {"yes", "no"} else "unknown",
                "confidence": confidence,
                "reason": reason,
            }
    except Exception:
        pass

    rec_match = re.search(r'"?recommend"?\s*[:=]\s*"?(yes|no)"?', clean, re.I)
    conf_match = re.search(r'"?confidence"?\s*[:=]\s*"?([0-9]*\.?[0-9]+%?)"?', clean, re.I)
    reason_match = re.search(r'"?reason"?\s*[:=]\s*"?(.*?)"?\s*(?:,|\n|$)', clean, re.I)

    recommend = rec_match.group(1).lower() if rec_match else "unknown"
    confidence = _normalize_confidence(conf_match.group(1)) if conf_match else -1.0
    reason = reason_match.group(1).strip() if reason_match else ""

    return {
        "recommend": recommend if recommend in {"yes", "no"} else "unknown",
        "confidence": confidence,
        "reason": reason,
    }


def parse_evidence_response(text: str) -> dict[str, Any]:
    raw = text.strip()
    clean = _strip_code_fence(raw)
    obj = _extract_json_object(clean)

    if obj is not None:
        recommend = _normalize_recommend(obj.get("recommend"))
        positive_evidence = _normalize_optional_unit_float(obj.get("positive_evidence"))
        negative_evidence = _normalize_optional_unit_float(obj.get("negative_evidence"))
        ambiguity = _normalize_optional_unit_float(obj.get("ambiguity"))
        missing_information = _normalize_optional_unit_float(obj.get("missing_information"))
        raw_confidence = _normalize_optional_unit_float(
            obj.get("raw_confidence", obj.get("confidence"))
        )
        reason = str(obj.get("reason", "")).strip()
    else:
        rec_match = re.search(r'"?recommend"?\s*[:=]\s*"?(yes|no)"?', clean, re.I)
        recommend = rec_match.group(1).lower() if rec_match else "unknown"
        positive_evidence = _normalize_optional_unit_float(
            _regex_number_field(clean, "positive_evidence")
        )
        negative_evidence = _normalize_optional_unit_float(
            _regex_number_field(clean, "negative_evidence")
        )
        ambiguity = _normalize_optional_unit_float(_regex_number_field(clean, "ambiguity"))
        missing_information = _normalize_optional_unit_float(
            _regex_number_field(clean, "missing_information")
        )
        raw_confidence = _normalize_optional_unit_float(
            _regex_number_field(clean, "raw_confidence")
            or _regex_number_field(clean, "confidence")
        )
        reason = _regex_field(clean, "reason") or ""

    evidence_values = [
        positive_evidence,
        negative_evidence,
        ambiguity,
        missing_information,
        raw_confidence,
    ]
    parse_success = recommend in {"yes", "no"} and all(value is not None for value in evidence_values)
    evidence_margin = (
        positive_evidence - negative_evidence
        if positive_evidence is not None and negative_evidence is not None
        else None
    )

    return {
        "recommend": recommend if recommend in {"yes", "no"} else "unknown",
        "positive_evidence": positive_evidence,
        "negative_evidence": negative_evidence,
        "evidence_margin": evidence_margin,
        "abs_evidence_margin": abs(evidence_margin) if evidence_margin is not None else None,
        "ambiguity": ambiguity,
        "missing_information": missing_information,
        "raw_confidence": raw_confidence,
        "reason": reason,
        "parse_success": parse_success,
        "parse_error": "" if parse_success else "missing_or_invalid_evidence_fields",
    }

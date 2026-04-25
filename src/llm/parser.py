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


def _normalize_item_id(value: Any) -> str:
    return str(value or "").strip()


def _normalize_rank(value: Any, fallback: int) -> int:
    try:
        rank = int(float(str(value).strip()))
    except (TypeError, ValueError):
        rank = fallback
    return max(1, rank)


def _normalize_recommended_items(obj: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = obj.get("recommended_items")
    if raw_items is None and "recommended_item_ids" in obj:
        raw_items = [
            {"candidate_item_id": item_id, "rank": idx + 1}
            for idx, item_id in enumerate(obj.get("recommended_item_ids") or [])
        ]

    if not isinstance(raw_items, list):
        return []

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items, start=1):
        if isinstance(item, dict):
            candidate_item_id = _normalize_item_id(
                item.get("candidate_item_id")
                or item.get("item_id")
                or item.get("id")
                or item.get("asin")
            )
            rank = _normalize_rank(item.get("rank"), fallback=idx)
            reason = str(item.get("reason", "")).strip()
            normalized_item = {
                "candidate_item_id": candidate_item_id,
                "rank": rank,
                "reason": reason,
            }
            for field in [
                "relevance_probability",
                "positive_evidence",
                "negative_evidence",
                "ambiguity",
                "missing_information",
            ]:
                normalized_item[field] = _normalize_optional_unit_float(item.get(field))
            normalized.append(normalized_item)
        else:
            normalized.append(
                {
                    "candidate_item_id": _normalize_item_id(item),
                    "rank": idx,
                    "reason": "",
                    "relevance_probability": None,
                    "positive_evidence": None,
                    "negative_evidence": None,
                    "ambiguity": None,
                    "missing_information": None,
                }
            )
    return normalized


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


def parse_relevance_evidence_response(text: str) -> dict[str, Any]:
    raw = text.strip()
    clean = _strip_code_fence(raw)
    obj = _extract_json_object(clean)

    if obj is not None:
        relevance_probability = _normalize_optional_unit_float(obj.get("relevance_probability"))
        positive_evidence = _normalize_optional_unit_float(obj.get("positive_evidence"))
        negative_evidence = _normalize_optional_unit_float(obj.get("negative_evidence"))
        ambiguity = _normalize_optional_unit_float(obj.get("ambiguity"))
        missing_information = _normalize_optional_unit_float(obj.get("missing_information"))
        recommend = _normalize_recommend(obj.get("recommend"))
        reason = str(obj.get("reason", "")).strip()
    else:
        relevance_probability = _normalize_optional_unit_float(
            _regex_number_field(clean, "relevance_probability")
        )
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
        rec_match = re.search(r'"?recommend"?\s*[:=]\s*"?(yes|no|true|false|recommend|reject)"?', clean, re.I)
        recommend = _normalize_recommend(rec_match.group(1) if rec_match else None)
        reason = _regex_field(clean, "reason") or ""

    evidence_values = [
        relevance_probability,
        positive_evidence,
        negative_evidence,
        ambiguity,
        missing_information,
    ]
    parse_success = recommend in {"yes", "no"} and all(value is not None for value in evidence_values)
    evidence_margin = (
        positive_evidence - negative_evidence
        if positive_evidence is not None and negative_evidence is not None
        else None
    )
    abs_evidence_margin = abs(evidence_margin) if evidence_margin is not None else None
    evidence_risk = (
        (1.0 - abs_evidence_margin + ambiguity + missing_information) / 3.0
        if abs_evidence_margin is not None
        and ambiguity is not None
        and missing_information is not None
        else None
    )

    return {
        "relevance_probability": relevance_probability,
        "positive_evidence": positive_evidence,
        "negative_evidence": negative_evidence,
        "evidence_margin": evidence_margin,
        "abs_evidence_margin": abs_evidence_margin,
        "ambiguity": ambiguity,
        "missing_information": missing_information,
        "evidence_risk": evidence_risk,
        "recommend": recommend if recommend in {"yes", "no"} else "unknown",
        "reason": reason,
        "parse_success": parse_success,
        "parse_error": "" if parse_success else "missing_or_invalid_relevance_evidence_fields",
    }


def parse_recommendation_list_plain_response(
    text: str,
    candidate_item_ids: list[str] | None = None,
) -> dict[str, Any]:
    raw = text.strip()
    clean = _strip_code_fence(raw)
    obj = _extract_json_object(clean)
    candidate_set = {str(item_id) for item_id in (candidate_item_ids or [])}

    if obj is None:
        return {
            "recommended_items": [],
            "selection_rationale": "",
            "parse_success": False,
            "schema_valid": False,
            "invalid_item_count": 0,
            "duplicate_item_count": 0,
            "parse_error": "missing_json_object",
        }

    items = _normalize_recommended_items(obj)
    seen: set[str] = set()
    duplicate_count = 0
    invalid_count = 0
    normalized_items: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda row: row["rank"]):
        item_id = item["candidate_item_id"]
        if not item_id:
            invalid_count += 1
            continue
        if candidate_set and item_id not in candidate_set:
            invalid_count += 1
        if item_id in seen:
            duplicate_count += 1
        seen.add(item_id)
        normalized_items.append(
            {
                "candidate_item_id": item_id,
                "rank": item["rank"],
                "reason": item.get("reason", ""),
            }
        )

    schema_valid = bool(normalized_items) and invalid_count == 0 and duplicate_count == 0
    return {
        "recommended_items": normalized_items,
        "selection_rationale": str(obj.get("selection_rationale", "")).strip(),
        "parse_success": schema_valid,
        "schema_valid": schema_valid,
        "invalid_item_count": int(invalid_count),
        "duplicate_item_count": int(duplicate_count),
        "parse_error": "" if schema_valid else "invalid_or_duplicate_recommended_items",
    }


def parse_recommendation_list_evidence_response(
    text: str,
    candidate_item_ids: list[str] | None = None,
) -> dict[str, Any]:
    raw = text.strip()
    clean = _strip_code_fence(raw)
    obj = _extract_json_object(clean)
    candidate_set = {str(item_id) for item_id in (candidate_item_ids or [])}

    if obj is None:
        return {
            "recommended_items": [],
            "global_uncertainty": None,
            "selection_rationale": "",
            "parse_success": False,
            "schema_valid": False,
            "invalid_item_count": 0,
            "duplicate_item_count": 0,
            "parse_error": "missing_json_object",
        }

    items = _normalize_recommended_items(obj)
    seen: set[str] = set()
    duplicate_count = 0
    invalid_count = 0
    evidence_missing_count = 0
    normalized_items: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda row: row["rank"]):
        item_id = item["candidate_item_id"]
        if not item_id:
            invalid_count += 1
            continue
        if candidate_set and item_id not in candidate_set:
            invalid_count += 1
        if item_id in seen:
            duplicate_count += 1
        seen.add(item_id)

        positive = item.get("positive_evidence")
        negative = item.get("negative_evidence")
        ambiguity = item.get("ambiguity")
        missing = item.get("missing_information")
        relevance = item.get("relevance_probability")
        values = [relevance, positive, negative, ambiguity, missing]
        if any(value is None for value in values):
            evidence_missing_count += 1
            margin = None
            abs_margin = None
            risk = None
        else:
            margin = positive - negative
            abs_margin = abs(margin)
            risk = (1.0 - abs_margin + ambiguity + missing) / 3.0
            risk = max(0.0, min(1.0, risk))

        normalized_items.append(
            {
                "candidate_item_id": item_id,
                "rank": item["rank"],
                "relevance_probability": relevance,
                "positive_evidence": positive,
                "negative_evidence": negative,
                "evidence_margin": margin,
                "abs_evidence_margin": abs_margin,
                "ambiguity": ambiguity,
                "missing_information": missing,
                "evidence_risk": risk,
                "reason": item.get("reason", ""),
            }
        )

    global_uncertainty = _normalize_optional_unit_float(obj.get("global_uncertainty"))
    schema_valid = (
        bool(normalized_items)
        and invalid_count == 0
        and duplicate_count == 0
        and evidence_missing_count == 0
        and global_uncertainty is not None
    )
    return {
        "recommended_items": normalized_items,
        "global_uncertainty": global_uncertainty,
        "selection_rationale": str(obj.get("selection_rationale", "")).strip(),
        "parse_success": schema_valid,
        "schema_valid": schema_valid,
        "invalid_item_count": int(invalid_count),
        "duplicate_item_count": int(duplicate_count),
        "evidence_missing_count": int(evidence_missing_count),
        "parse_error": "" if schema_valid else "invalid_or_missing_evidence_recommended_items",
    }

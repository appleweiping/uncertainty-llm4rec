from __future__ import annotations

import json
import re
from typing import Any


def strip_thinking_blocks(text: str) -> str:
    """Remove model-visible reasoning wrappers before task-specific parsing."""
    clean = str(text or "")
    clean = re.sub(r"<think\b[^>]*>.*?</think>", "", clean, flags=re.I | re.S)
    clean = re.sub(r"^\s*</think>\s*", "", clean, flags=re.I)
    return clean.strip()


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
        confidence_labels = {
            "very low": 0.1,
            "low": 0.2,
            "medium": 0.5,
            "mid": 0.5,
            "moderate": 0.5,
            "high": 0.8,
            "very high": 0.9,
        }
        normalized_text = re.sub(r"[\s_\-]+", " ", text.lower()).strip()
        if normalized_text in confidence_labels:
            conf = confidence_labels[normalized_text]
        else:
            conf = float(text)

    if conf > 1.0:
        conf = conf / 100.0

    conf = max(0.0, min(1.0, conf))
    return conf


def _extract_json_payload(clean: str) -> dict[str, Any] | None:
    try:
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(clean[start:end])
    except Exception:
        return None
    return None


def _normalize_item_id_list(value: Any) -> list[str]:
    normalized: list[str] = []

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                item_id = str(item.get("item_id") or item.get("id") or "").strip()
            else:
                item_id = str(item).strip()
            if item_id:
                normalized.append(item_id)
        return normalized

    if isinstance(value, str):
        for token in re.split(r"[\s,\[\]\(\)\|]+", value):
            token = token.strip().strip('"').strip("'")
            if token:
                normalized.append(token)
    return normalized


def _extract_allowed_mentions(clean: str, allowed_item_ids: list[str]) -> list[str]:
    matches: list[tuple[int, str]] = []
    for item_id in allowed_item_ids:
        for match in re.finditer(rf"\b{re.escape(item_id)}\b", clean):
            matches.append((match.start(), item_id))

    matches.sort(key=lambda pair: pair[0])
    ordered: list[str] = []
    for _, item_id in matches:
        if item_id not in ordered:
            ordered.append(item_id)
    return ordered


def _extract_pairwise_text_preference(
    clean: str,
    *,
    item_a_id: str | None = None,
    item_b_id: str | None = None,
) -> tuple[str, bool]:
    candidates: dict[str, str] = {}
    if item_a_id:
        candidates[str(item_a_id)] = str(item_a_id)
        candidates["A"] = str(item_a_id)
    if item_b_id:
        candidates[str(item_b_id)] = str(item_b_id)
        candidates["B"] = str(item_b_id)

    if not candidates:
        return "", True

    ordered_tokens = sorted(candidates.keys(), key=len, reverse=True)
    token_pattern = "|".join(re.escape(token) for token in ordered_tokens)
    preference_patterns = [
        rf"(?:prefer|preferred|choose|chose|pick|picked|select|selected|recommend|recommended|go with|lean toward|favor|favors)\s+(?:candidate\s+)?({token_pattern})\b",
        rf"(?:candidate\s+)?({token_pattern})\b\s+(?:is|looks|seems)\s+(?:better|preferred|more suitable|the better choice|the safer choice|the winner)\b",
        rf"(?:winner|best choice|top choice)\s*(?:is|:)\s*(?:candidate\s+)?({token_pattern})\b",
    ]

    resolved: list[str] = []
    for pattern in preference_patterns:
        for match in re.finditer(pattern, clean, re.I):
            raw_token = match.group(1).strip()
            token = raw_token.upper()
            normalized = candidates.get(token) or candidates.get(raw_token)
            if normalized:
                resolved.append(normalized)

    unique_resolved = list(dict.fromkeys(resolved))
    if len(unique_resolved) == 1:
        return unique_resolved[0], False
    if len(unique_resolved) > 1:
        return "", True
    return "", True


def parse_pointwise_response(text: str) -> dict[str, Any]:
    raw = strip_thinking_blocks(text)
    clean = _strip_code_fence(raw)
    obj = _extract_json_payload(clean)

    if obj is not None:
        recommend = str(obj.get("recommend", "unknown")).strip().lower()
        confidence = _normalize_confidence(obj.get("confidence", -1.0))
        reason = str(obj.get("reason", "")).strip()
        return {
            "recommend": recommend if recommend in {"yes", "no"} else "unknown",
            "confidence": confidence,
            "reason": reason,
        }

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


def parse_candidate_ranking_response(
    text: str,
    *,
    allowed_item_ids: list[str] | None = None,
    topk: int | None = None,
) -> dict[str, Any]:
    raw = strip_thinking_blocks(text)
    clean = _strip_code_fence(raw)
    allowed_item_ids = [str(item_id).strip() for item_id in (allowed_item_ids or []) if str(item_id).strip()]

    ranked_item_ids: list[str] = []
    topk_item_ids: list[str] = []
    confidence = -1.0
    reason = ""
    parse_mode = "failed"

    obj = _extract_json_payload(clean)
    if obj is not None:
        ranked_item_ids = _normalize_item_id_list(
            obj.get("ranked_item_ids")
            or obj.get("ranked_items")
            or obj.get("ranking")
            or []
        )
        topk_item_ids = _normalize_item_id_list(
            obj.get("topk_item_ids")
            or obj.get("topk")
            or []
        )
        confidence = _normalize_confidence(obj.get("confidence", -1.0))
        reason = str(obj.get("reason", "")).strip()
        parse_mode = "json"
    else:
        ranked_match = re.search(r'"?(ranked_item_ids|ranking|ranked_items)"?\s*[:=]\s*(\[.*?\])', clean, re.I | re.S)
        topk_match = re.search(r'"?(topk_item_ids|topk)"?\s*[:=]\s*(\[.*?\])', clean, re.I | re.S)
        confidence_match = re.search(r'"?confidence"?\s*[:=]\s*"?([0-9]*\.?[0-9]+%?)"?', clean, re.I)
        reason_match = re.search(r'"?reason"?\s*[:=]\s*"?(.*?)"?\s*(?:,|\n|$)', clean, re.I)

        if ranked_match:
            ranked_item_ids = _normalize_item_id_list(ranked_match.group(2))
            parse_mode = "regex_list"
        if topk_match:
            topk_item_ids = _normalize_item_id_list(topk_match.group(2))
            parse_mode = "regex_list"

        confidence = _normalize_confidence(confidence_match.group(1)) if confidence_match else -1.0
        reason = reason_match.group(1).strip() if reason_match else ""

    if not ranked_item_ids and allowed_item_ids:
        ordered_mentions = _extract_allowed_mentions(clean, allowed_item_ids)
        if ordered_mentions:
            ranked_item_ids = ordered_mentions
            parse_mode = "text_mentions"

    if not topk_item_ids and ranked_item_ids:
        topk_item_ids = ranked_item_ids[: topk] if topk is not None else list(ranked_item_ids)

    out_of_candidate_item_ids = []
    if allowed_item_ids:
        allowed_set = set(allowed_item_ids)
        for item_id in ranked_item_ids + topk_item_ids:
            if item_id not in allowed_set and item_id not in out_of_candidate_item_ids:
                out_of_candidate_item_ids.append(item_id)

    unique_ranked_item_ids: list[str] = []
    for item_id in ranked_item_ids:
        if item_id not in unique_ranked_item_ids:
            unique_ranked_item_ids.append(item_id)
    ranked_item_ids = unique_ranked_item_ids

    unique_topk_item_ids: list[str] = []
    for item_id in topk_item_ids:
        if item_id not in unique_topk_item_ids:
            unique_topk_item_ids.append(item_id)
    topk_item_ids = unique_topk_item_ids

    parse_success = bool(topk_item_ids or ranked_item_ids) and len(out_of_candidate_item_ids) == 0

    return {
        "ranked_item_ids": ranked_item_ids,
        "topk_item_ids": topk_item_ids,
        "confidence": confidence,
        "reason": reason,
        "parse_mode": parse_mode,
        "parse_success": parse_success,
        "out_of_candidate_item_ids": out_of_candidate_item_ids,
        "contains_out_of_candidate_item": len(out_of_candidate_item_ids) > 0,
    }


def parse_pairwise_preference_response(
    text: str,
    *,
    item_a_id: str | None = None,
    item_b_id: str | None = None,
) -> dict[str, Any]:
    raw = strip_thinking_blocks(text)
    clean = _strip_code_fence(raw)

    preferred_item = ""
    confidence = -1.0
    reason = ""
    parse_mode = "failed"

    obj = _extract_json_payload(clean)
    if obj is not None:
        preferred_item = str(
            obj.get("preferred_item")
            or obj.get("winner")
            or obj.get("preference")
            or ""
        ).strip()
        confidence = _normalize_confidence(obj.get("confidence", -1.0))
        reason = str(obj.get("reason", "")).strip()
        parse_mode = "json"
    else:
        pref_match = re.search(r'"?(preferred_item|winner|preference)"?\s*[:=]\s*"?(A|B|[A-Za-z0-9_\-]+)"?', clean, re.I)
        confidence_match = re.search(r'"?confidence"?\s*[:=]\s*"?([0-9]*\.?[0-9]+%?)"?', clean, re.I)
        reason_match = re.search(r'"?reason"?\s*[:=]\s*"?(.*?)"?\s*(?:,|\n|$)', clean, re.I)
        if pref_match:
            preferred_item = pref_match.group(2).strip()
            parse_mode = "regex"
        confidence = _normalize_confidence(confidence_match.group(1)) if confidence_match else -1.0
        reason = reason_match.group(1).strip() if reason_match else ""

    if preferred_item == "":
        preferred_item, ambiguous_from_text = _extract_pairwise_text_preference(
            clean,
            item_a_id=item_a_id,
            item_b_id=item_b_id,
        )
        if preferred_item:
            parse_mode = "text_preference"
        elif ambiguous_from_text:
            parse_mode = "failed"

    if confidence < 0.0:
        freeform_confidence_match = re.search(
            r"([0-9]*\.?[0-9]+%?)\s*(?:confidence|confident)\b|\bconfidence\s+(?:is|=|:)\s*([0-9]*\.?[0-9]+%?)",
            clean,
            re.I,
        )
        if freeform_confidence_match:
            confidence = _normalize_confidence(
                freeform_confidence_match.group(1) or freeform_confidence_match.group(2)
            )

    normalized_preferred_item = preferred_item
    if preferred_item.upper() == "A" and item_a_id:
        normalized_preferred_item = str(item_a_id)
    elif preferred_item.upper() == "B" and item_b_id:
        normalized_preferred_item = str(item_b_id)

    allowed_item_ids = {str(item_a_id).strip(), str(item_b_id).strip()} - {""}
    ambiguous_preference = normalized_preferred_item == ""
    if normalized_preferred_item and allowed_item_ids and normalized_preferred_item not in allowed_item_ids:
        ambiguous_preference = True

    if not reason and parse_mode == "text_preference":
        reason = clean[:240].strip()

    parse_success = not ambiguous_preference and normalized_preferred_item != ""

    return {
        "preferred_item": normalized_preferred_item if parse_success else "",
        "confidence": confidence,
        "reason": reason,
        "parse_mode": parse_mode,
        "parse_success": parse_success,
        "ambiguous_preference": ambiguous_preference,
    }


def parse_response(text: str) -> dict[str, Any]:
    return parse_pointwise_response(text)

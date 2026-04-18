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


def _normalize_score(value: Any) -> float:
    if value is None:
        return -1.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return -1.0
    return float(text)


def _coerce_item_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = re.split(r"[\n,>]+", value)
        return [part.strip().strip('"').strip("'") for part in parts if part.strip()]
    return []


def _extract_json_object(clean: str) -> dict[str, Any] | None:
    try:
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start != -1 and end > start:
            obj = json.loads(clean[start:end])
            if isinstance(obj, dict):
                return obj
    except Exception:
        return None
    return None


def parse_candidate_score_list_response(text: str) -> dict[str, Any]:
    raw = text.strip()
    clean = _strip_code_fence(raw)

    obj = _extract_json_object(clean) or {}
    scores = (
        obj.get("scores")
        or obj.get("candidate_scores")
        or obj.get("items")
        or []
    )

    parsed_scores: list[dict[str, Any]] = []
    if isinstance(scores, list):
        for entry in scores:
            if not isinstance(entry, dict):
                continue
            item_id = str(
                entry.get("item_id")
                or entry.get("candidate_item_id")
                or entry.get("id")
                or ""
            ).strip()
            if not item_id:
                continue
            parsed_scores.append(
                {
                    "item_id": item_id,
                    "score": _normalize_score(entry.get("score", entry.get("confidence", -1.0))),
                    "reason": str(entry.get("reason", "")).strip(),
                }
            )

    if not parsed_scores:
        pattern = re.compile(
            r"(?:item_id|candidate_item_id|id)\s*[:=]\s*\"?([^\s,\"}]+)\"?.{0,80}?(?:score|confidence)\s*[:=]\s*\"?([0-9]*\.?[0-9]+)\"?",
            re.I,
        )
        for item_id, score in pattern.findall(clean):
            parsed_scores.append(
                {
                    "item_id": str(item_id).strip(),
                    "score": _normalize_score(score),
                    "reason": "",
                }
            )

    parsed_scores = sorted(parsed_scores, key=lambda row: row.get("score", -1.0), reverse=True)
    ranked_item_ids = [row["item_id"] for row in parsed_scores]
    selected_item_id = str(
        obj.get("selected_item_id")
        or obj.get("top_1_item_id")
        or (ranked_item_ids[0] if ranked_item_ids else "")
    ).strip()

    return {
        "ranking_mode": "score_list",
        "candidate_scores": parsed_scores,
        "ranked_item_ids": ranked_item_ids,
        "top_k_item_ids": ranked_item_ids,
        "selected_item_id": selected_item_id,
        "reason": str(obj.get("reason", "")).strip(),
    }


def parse_candidate_rank_topk_response(text: str) -> dict[str, Any]:
    raw = text.strip()
    clean = _strip_code_fence(raw)

    obj = _extract_json_object(clean) or {}
    ranked_item_ids = _coerce_item_list(
        obj.get("ranked_item_ids")
        or obj.get("ranking")
        or obj.get("ranked_list")
    )
    top_k_item_ids = _coerce_item_list(
        obj.get("top_k_item_ids")
        or obj.get("topk_item_ids")
        or obj.get("selected_item_ids")
        or ranked_item_ids
    )

    if not ranked_item_ids:
        matches = re.findall(r"(?:item_id|candidate_item_id|id)\s*[:=]\s*\"?([^\s,\"}]+)\"?", clean, re.I)
        ranked_item_ids = [str(item).strip() for item in matches if str(item).strip()]
        if not top_k_item_ids:
            top_k_item_ids = ranked_item_ids

    selected_item_id = str(
        obj.get("selected_item_id")
        or obj.get("top_1_item_id")
        or (ranked_item_ids[0] if ranked_item_ids else "")
    ).strip()

    return {
        "ranking_mode": "rank_topk",
        "candidate_scores": [],
        "ranked_item_ids": ranked_item_ids,
        "top_k_item_ids": top_k_item_ids,
        "selected_item_id": selected_item_id,
        "reason": str(obj.get("reason", "")).strip(),
    }


def parse_ranking_response(text: str, ranking_mode: str = "score_list") -> dict[str, Any]:
    mode = str(ranking_mode).strip().lower()
    if mode == "rank_topk":
        return parse_candidate_rank_topk_response(text)
    return parse_candidate_score_list_response(text)

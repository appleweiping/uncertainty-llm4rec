"""Parse LLM preference JSON (listwise / pairwise); map labels to item IDs; fail closed."""

from __future__ import annotations

import json
import re
from typing import Any


def _sanitize_json_blob(blob: str) -> str:
    """Undo common invalid JSON escapes from LLMs (\"'\" inside double-quoted strings)."""
    if not blob:
        return blob
    return blob.replace("\\'", "'")


def _extract_json_blob(raw: str) -> str | None:
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return None


def parse_listwise_response(
    raw_output: str,
    *,
    label_to_item_id: dict[str, str],
    allow_duplicate_labels: bool = False,
) -> dict[str, Any]:
    """Fail closed on invalid structure or unknown labels."""
    blob = _extract_json_blob(raw_output)
    if not blob:
        return {"ok": False, "error": "no_json", "raw_output": raw_output, "ranking": [], "uncertainty": {}}
    blob = _sanitize_json_blob(blob)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"json:{exc}", "raw_output": raw_output, "ranking": [], "uncertainty": {}}
    ranking = data.get("ranking")
    if not isinstance(ranking, list):
        return {"ok": False, "error": "missing_ranking", "raw_output": raw_output, "ranking": [], "uncertainty": {}}
    seen: set[str] = set()
    out_rank: list[dict[str, Any]] = []
    for row in ranking:
        if not isinstance(row, dict):
            continue
        lab = str(row.get("label", "")).strip()
        if lab not in label_to_item_id:
            return {"ok": False, "error": f"unknown_label:{lab}", "raw_output": raw_output, "ranking": [], "uncertainty": {}}
        if lab in seen and not allow_duplicate_labels:
            return {"ok": False, "error": f"duplicate_label:{lab}", "raw_output": raw_output, "ranking": [], "uncertainty": {}}
        seen.add(lab)
        item_id = label_to_item_id[lab]
        out_rank.append(
            {
                "label": lab,
                "item_id": item_id,
                "score": float(row.get("score", 0.0) or 0.0),
                "confidence": float(row.get("confidence", 0.0) or 0.0),
                "reason": str(row.get("reason", "")),
            }
        )
    unc = data.get("uncertainty") if isinstance(data.get("uncertainty"), dict) else {}
    return {
        "ok": True,
        "error": None,
        "raw_output": raw_output,
        "ranking": out_rank,
        "uncertainty": unc,
    }


def parse_pairwise_response(
    raw_output: str,
    *,
    label_to_item_id: dict[str, str],
) -> dict[str, Any]:
    blob = _extract_json_blob(raw_output)
    if not blob:
        return {"ok": False, "error": "no_json", "raw_output": raw_output, "pairs": []}
    blob = _sanitize_json_blob(blob)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"json:{exc}", "raw_output": raw_output, "pairs": []}
    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        return {"ok": False, "error": "missing_pairs", "raw_output": raw_output, "pairs": []}
    out: list[dict[str, Any]] = []
    for row in pairs:
        if not isinstance(row, dict):
            continue
        left = str(row.get("left", "")).strip()
        right = str(row.get("right", "")).strip()
        win = str(row.get("winner", "")).strip()
        if left not in label_to_item_id or right not in label_to_item_id:
            return {"ok": False, "error": "unknown_label", "raw_output": raw_output, "pairs": []}
        if win not in (left, right):
            return {"ok": False, "error": "bad_winner", "raw_output": raw_output, "pairs": []}
        out.append(
            {
                "left": left,
                "right": right,
                "left_item_id": label_to_item_id[left],
                "right_item_id": label_to_item_id[right],
                "winner": win,
                "winner_item_id": label_to_item_id[win],
                "confidence": float(row.get("confidence", 0.0) or 0.0),
                "reason": str(row.get("reason", "")),
            }
        )
    oc = data.get("overall_confidence")
    return {
        "ok": True,
        "error": None,
        "raw_output": raw_output,
        "pairs": out,
        "overall_confidence": float(oc) if isinstance(oc, (int, float)) else None,
    }

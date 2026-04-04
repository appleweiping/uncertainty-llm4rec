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
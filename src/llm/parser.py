# src/llm/parser.py
from __future__ import annotations

import json
import re
from typing import Any


def parse_response(text: str) -> dict[str, Any]:
    text = text.strip()

    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            obj = json.loads(text[start:end])
            recommend = str(obj.get("recommend", "unknown")).strip().lower()
            confidence = float(obj.get("confidence", -1.0))
            reason = str(obj.get("reason", "")).strip()
            return {
                "recommend": recommend,
                "confidence": max(0.0, min(1.0, confidence)) if confidence >= 0 else -1.0,
                "reason": reason,
            }
    except Exception:
        pass

    rec_match = re.search(r'"?recommend"?\s*:\s*"?(yes|no)"?', text, re.I)
    conf_match = re.search(r'"?confidence"?\s*:\s*([0-9]*\.?[0-9]+)', text, re.I)

    recommend = rec_match.group(1).lower() if rec_match else "unknown"
    confidence = float(conf_match.group(1)) if conf_match else -1.0

    return {
        "recommend": recommend,
        "confidence": max(0.0, min(1.0, confidence)) if confidence >= 0 else -1.0,
        "reason": "",
    }
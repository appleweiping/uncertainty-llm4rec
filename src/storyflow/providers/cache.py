"""Small file cache for API observation responses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def build_cache_key(
    *,
    provider: str,
    model: str,
    prompt_template: str,
    temperature: float,
    input_hash: str,
    max_tokens: int | None = None,
    request_options: dict[str, Any] | None = None,
    execution_mode: str | None = None,
) -> str:
    payload = {
        "provider": provider,
        "model": model,
        "prompt_template": prompt_template,
        "temperature": float(temperature),
        "input_hash": input_hash,
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if request_options:
        payload["request_options"] = request_options
    if execution_mode:
        payload["execution_mode"] = execution_mode
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ResponseCache:
    """JSON response cache keyed by provider/model/template/hash settings."""

    def __init__(self, cache_dir: str | Path, *, enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

    def _path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def get(self, cache_key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path(cache_key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, cache_key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._path(cache_key).write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

"""File-backed response cache scaffold."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ResponseCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def key_for(self, *, provider: str, model: str, prompt: str, params: dict[str, Any] | None = None) -> str:
        payload = {
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "params": params or {},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, value: dict[str, Any]) -> Path:
        path = self._path(key)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return path

    def _path(self, key: str) -> Path:
        if not key or any(char in key for char in "\\/"):
            raise ValueError("cache key must be a non-empty filename-safe string")
        return self.cache_dir / f"{key}.json"

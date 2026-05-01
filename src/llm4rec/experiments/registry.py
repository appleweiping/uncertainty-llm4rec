"""Method registry for experiment runners."""

from __future__ import annotations

from typing import Any, Callable

PredictionFn = Callable[[dict[str, Any], list[dict[str, Any]]], list[dict[str, Any]]]

_METHODS: dict[str, PredictionFn] = {}


def register_method(name: str, func: PredictionFn) -> None:
    if not name.strip():
        raise ValueError("method name must be non-empty")
    _METHODS[name] = func


def get_method(name: str) -> PredictionFn:
    try:
        return _METHODS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_METHODS)) or "<none>"
        raise ValueError(f"unknown method {name!r}; known: {known}") from exc

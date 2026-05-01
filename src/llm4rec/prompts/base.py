"""Prompt contracts for generative recommendation observation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class PromptBuildResult:
    prompt: str
    prompt_template_id: str
    prompt_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def make_prompt_result(
    *,
    prompt: str,
    prompt_template_id: str,
    metadata: dict[str, Any] | None = None,
) -> PromptBuildResult:
    prompt_hash = hash_prompt(prompt)
    payload = {
        "prompt_template_id": prompt_template_id,
        "prompt_hash": prompt_hash,
        **(metadata or {}),
    }
    return PromptBuildResult(
        prompt=prompt,
        prompt_template_id=prompt_template_id,
        prompt_hash=prompt_hash,
        metadata=payload,
    )

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.io import load_jsonl as load_jsonl_rows
from src.utils.io import save_jsonl as save_jsonl_rows


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class FunctionPromptBuilderAdapter:
    def __init__(self, fn, template_path: str | Path):
        self.fn = fn
        self.template_path = str(template_path)

    def build_pointwise_prompt(self, sample: dict[str, Any], candidate: dict[str, Any]) -> str:
        try:
            return self.fn(sample, candidate, template_path=self.template_path)
        except TypeError:
            return self.fn(sample, candidate)


def get_prompt_builder(prompt_path: str | Path):
    try:
        from src.llm.prompt_builder import PromptBuilder

        return PromptBuilder(template_path=str(prompt_path))
    except Exception:
        try:
            from src.llm.prompt_builder import build_pointwise_prompt

            return FunctionPromptBuilderAdapter(build_pointwise_prompt, template_path=prompt_path)
        except Exception as exc:
            raise ImportError("Cannot find a usable prompt builder in src/llm/prompt_builder.py") from exc


def load_jsonl(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows = load_jsonl_rows(path)
    if max_samples is not None and max_samples > 0:
        return rows[:max_samples]
    return rows


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    save_jsonl_rows(records, path)

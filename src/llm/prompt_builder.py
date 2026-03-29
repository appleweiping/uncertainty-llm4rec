# src/llm/prompt_builder.py
from __future__ import annotations

from pathlib import Path
from typing import Any


class PromptBuilder:
    def __init__(self, template_path: str) -> None:
        self.template = Path(template_path).read_text(encoding="utf-8")

    @staticmethod
    def build_history_block(history_items: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for idx, item in enumerate(history_items, start=1):
            title = item.get("title", "")
            meta = item.get("meta", "")
            if meta:
                lines.append(f"{idx}. {title} | {meta}")
            else:
                lines.append(f"{idx}. {title}")
        return "\n".join(lines)

    def build_pointwise_prompt(
        self,
        sample: dict[str, Any],
        candidate: dict[str, Any],
    ) -> str:
        return self.template.format(
            history_block=self.build_history_block(sample["history_items"]),
            candidate_title=candidate.get("title", ""),
            candidate_meta=candidate.get("meta", ""),
        )
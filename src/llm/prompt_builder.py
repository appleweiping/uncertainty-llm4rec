from __future__ import annotations

from pathlib import Path
from typing import Any


class PromptBuilder:
    def __init__(self, template_path: str) -> None:
        self.template = Path(template_path).read_text(encoding="utf-8")

    @staticmethod
    def build_history_block(sample: dict[str, Any]) -> str:
        if "history_items" in sample and isinstance(sample["history_items"], list):
            lines: list[str] = []
            for idx, item in enumerate(sample["history_items"], start=1):
                title = str(item.get("title", "")).strip()
                meta = str(item.get("meta", "")).strip()
                text = f"{title} | {meta}".strip(" |")
                lines.append(f"{idx}. {text}" if text else f"{idx}. [EMPTY]")
            return "\n".join(lines)

        if "history" in sample:
            history = sample["history"]

            if isinstance(history, list):
                lines: list[str] = []
                for idx, item in enumerate(history, start=1):
                    if isinstance(item, dict):
                        title = str(item.get("title", "")).strip()
                        meta = str(item.get("meta", "")).strip()
                        text = str(item.get("text", "")).strip()
                        value = f"{title} | {meta}".strip(" |") or text or "[EMPTY]"
                        lines.append(f"{idx}. {value}")
                    else:
                        value = str(item).strip() or "[EMPTY]"
                        lines.append(f"{idx}. {value}")
                return "\n".join(lines)

            if isinstance(history, str):
                return history.strip()

        if "history_text" in sample and isinstance(sample["history_text"], str):
            return sample["history_text"].strip()

        return ""

    @staticmethod
    def build_candidate_fields(
        sample: dict[str, Any],
        candidate: dict[str, Any],
    ) -> tuple[str, str]:
        candidate_title = str(
            candidate.get("title")
            or sample.get("candidate_title")
            or ""
        ).strip()

        candidate_meta = str(
            candidate.get("meta")
            or sample.get("candidate_meta")
            or sample.get("candidate_description")
            or sample.get("candidate_text")
            or ""
        ).strip()

        if not candidate_title:
            candidate_title = str(
                sample.get("candidate_item_id")
                or candidate.get("item_id")
                or "Unknown Item"
            ).strip()

        return candidate_title, candidate_meta

    def build_pointwise_prompt(
        self,
        sample: dict[str, Any],
        candidate: dict[str, Any],
    ) -> str:
        history_block = self.build_history_block(sample)
        candidate_title, candidate_meta = self.build_candidate_fields(sample, candidate)

        return self.template.format(
            history_block=history_block,
            candidate_title=candidate_title,
            candidate_meta=candidate_meta,
        )
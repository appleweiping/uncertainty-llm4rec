from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_template(
    *,
    template_text: str | None = None,
    template_path: str | Path | None = None,
) -> str:
    if template_text is not None:
        return str(template_text)
    if template_path is None:
        raise ValueError("Either template_text or template_path must be provided.")
    return Path(template_path).read_text(encoding="utf-8")


def _stringify_history_item(item: Any) -> str:
    if isinstance(item, dict):
        title = str(item.get("title", "")).strip()
        meta = str(item.get("meta", "")).strip()
        text = str(item.get("text", "")).strip()
        return f"{title} | {meta}".strip(" |") or text or "[EMPTY]"
    return str(item).strip() or "[EMPTY]"


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
                return "\n".join(
                    f"{idx}. {_stringify_history_item(item)}"
                    for idx, item in enumerate(history, start=1)
                )

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

    @staticmethod
    def build_candidate_ranking_block(sample: dict[str, Any]) -> str:
        item_ids = sample.get("candidate_item_ids", [])
        titles = sample.get("candidate_titles", [])
        texts = sample.get("candidate_texts", [])
        groups = sample.get("candidate_popularity_groups", [])

        if not isinstance(item_ids, list) or len(item_ids) == 0:
            raise ValueError("Candidate ranking sample is missing candidate_item_ids.")

        lines: list[str] = []
        for idx, item_id in enumerate(item_ids, start=1):
            title = str(titles[idx - 1]).strip() if idx - 1 < len(titles) else ""
            text = str(texts[idx - 1]).strip() if idx - 1 < len(texts) else ""
            group = str(groups[idx - 1]).strip() if idx - 1 < len(groups) else "mid"

            content = text or (f"Title: {title}" if title else "")
            lines.append(
                f"{idx}. item_id={item_id}\n"
                f"   title={title or '[EMPTY]'}\n"
                f"   text={content or '[EMPTY]'}\n"
                f"   popularity_group={group or 'mid'}"
            )
        return "\n".join(lines)

    def build_pointwise_prompt(
        self,
        sample: dict[str, Any],
        candidate: dict[str, Any],
    ) -> str:
        return build_pointwise_prompt(sample, candidate, template_text=self.template)

    def build_candidate_ranking_prompt(
        self,
        sample: dict[str, Any],
        topk: int | None = None,
    ) -> str:
        return build_candidate_ranking_prompt(sample, topk=topk, template_text=self.template)

    def build_pairwise_preference_prompt(self, sample: dict[str, Any]) -> str:
        return build_pairwise_preference_prompt(sample, template_text=self.template)


def build_pointwise_prompt(
    sample: dict[str, Any],
    candidate: dict[str, Any],
    *,
    template_text: str | None = None,
    template_path: str | Path | None = None,
) -> str:
    template = _load_template(template_text=template_text, template_path=template_path)
    history_block = PromptBuilder.build_history_block(sample)
    candidate_title, candidate_meta = PromptBuilder.build_candidate_fields(sample, candidate)

    return template.format(
        history_block=history_block,
        candidate_title=candidate_title,
        candidate_meta=candidate_meta,
    )


def build_candidate_ranking_prompt(
    sample: dict[str, Any],
    *,
    topk: int | None = None,
    template_text: str | None = None,
    template_path: str | Path | None = None,
) -> str:
    template = _load_template(template_text=template_text, template_path=template_path)
    history_block = PromptBuilder.build_history_block(sample)
    candidate_block = PromptBuilder.build_candidate_ranking_block(sample)
    candidate_item_ids = [str(item_id).strip() for item_id in sample.get("candidate_item_ids", [])]
    if not candidate_item_ids:
        raise ValueError("Candidate ranking sample is missing candidate_item_ids.")

    resolved_topk = int(topk) if topk is not None else len(candidate_item_ids)
    resolved_topk = max(1, min(resolved_topk, len(candidate_item_ids)))

    return template.format(
        history_block=history_block,
        candidate_block=candidate_block,
        candidate_count=len(candidate_item_ids),
        allowed_item_ids=", ".join(candidate_item_ids),
        topk=resolved_topk,
    )


def build_pairwise_preference_prompt(
    sample: dict[str, Any],
    *,
    template_text: str | None = None,
    template_path: str | Path | None = None,
) -> str:
    template = _load_template(template_text=template_text, template_path=template_path)
    history_block = PromptBuilder.build_history_block(sample)

    item_a_id = str(sample.get("item_a_id", "")).strip()
    item_a_title = str(sample.get("item_a_title", "")).strip()
    item_a_text = str(sample.get("item_a_text", "")).strip()
    item_b_id = str(sample.get("item_b_id", "")).strip()
    item_b_title = str(sample.get("item_b_title", "")).strip()
    item_b_text = str(sample.get("item_b_text", "")).strip()

    if item_a_id == "" or item_b_id == "":
        raise ValueError("Pairwise preference sample is missing item_a_id or item_b_id.")

    return template.format(
        history_block=history_block,
        item_a_id=item_a_id,
        item_a_title=item_a_title or "[EMPTY]",
        item_a_text=item_a_text or "[EMPTY]",
        item_b_id=item_b_id,
        item_b_title=item_b_title or "[EMPTY]",
        item_b_text=item_b_text or "[EMPTY]",
    )

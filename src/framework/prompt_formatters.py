from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RANKING_TEMPLATE = Path("prompts/framework/qwen_candidate_ranking_baseline.txt")
RELEVANCE_TEMPLATE = Path("prompts/framework/qwen_candidate_relevance_baseline.txt")


def _read_template(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt template not found: {p}")
    return p.read_text(encoding="utf-8").strip()


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def format_ranking_prompt(sample: dict[str, Any], template_path: str | Path = RANKING_TEMPLATE) -> tuple[str, str]:
    """Format a closed-candidate ranking sample.

    The target remains a plain JSON string and contains no confidence,
    evidence, or calibrated-probability fields.
    """
    template = _read_template(template_path)
    model_input = {
        "user_history": sample["input"].get("user_history", []),
        "candidate_pool": sample["input"].get("candidate_pool", []),
    }
    target = {
        "ranked_item_ids": sample["output"].get("ranked_item_ids", []),
    }
    prompt = f"{template}\n\nInput JSON:\n{_compact_json(model_input)}\n\nOutput JSON:\n"
    return prompt, _compact_json(target)


def format_relevance_prompt(sample: dict[str, Any], template_path: str | Path = RELEVANCE_TEMPLATE) -> tuple[str, str]:
    """Format a pointwise relevance sample.

    The target is a raw relevance label for baseline training only. CEP
    calibration remains a later valid-set post-processing step.
    """
    template = _read_template(template_path)
    model_input = {
        "user_history": sample["input"].get("user_history", []),
        "candidate_item": sample["input"].get("candidate_item", {}),
    }
    target = {
        "relevance_label": int(sample["output"].get("relevance_label", 0)),
    }
    prompt = f"{template}\n\nInput JSON:\n{_compact_json(model_input)}\n\nOutput JSON:\n"
    return prompt, _compact_json(target)


def format_sample(
    sample: dict[str, Any],
    task_type: str,
    template_path: str | Path | None = None,
) -> tuple[str, str]:
    if task_type == "candidate_ranking_listwise":
        return format_ranking_prompt(sample, template_path or RANKING_TEMPLATE)
    if task_type == "candidate_relevance_pointwise":
        return format_relevance_prompt(sample, template_path or RELEVANCE_TEMPLATE)
    raise ValueError(f"Unsupported task_type: {task_type}")


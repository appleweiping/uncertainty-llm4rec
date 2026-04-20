from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.llm.prompt_builder import build_candidate_ranking_prompt
from src.utils.exp_io import load_jsonl


@dataclass(frozen=True)
class RankSupervisedExample:
    source_event_id: str
    user_id: str
    prompt: str
    target_text: str
    positive_item_id: str
    candidate_item_ids: list[str]


def load_rank_samples(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    return load_jsonl(path, max_samples=max_samples)


def _normalize_target_ranking(sample: dict[str, Any], topk: int) -> list[str]:
    candidate_ids = [str(item_id) for item_id in sample.get("candidate_item_ids", [])]
    positive_item_id = str(sample.get("positive_item_id", "")).strip()
    if not candidate_ids:
        raise ValueError("Ranking training sample is missing candidate_item_ids.")
    if positive_item_id and positive_item_id in candidate_ids:
        ordered = [positive_item_id] + [item_id for item_id in candidate_ids if item_id != positive_item_id]
    else:
        ordered = candidate_ids[:]
    return ordered[:topk]


def _build_target_text(
    sample: dict[str, Any],
    *,
    topk: int,
    include_reason: bool,
) -> str:
    ranked_item_ids = _normalize_target_ranking(sample, topk=topk)
    payload: dict[str, Any] = {
        "ranked_item_ids": ranked_item_ids,
        "topk_item_ids": ranked_item_ids,
        "confidence": 1.0,
    }
    if include_reason:
        payload["reason"] = "The positive item should appear before other candidates in the supervised target ranking."
    return json.dumps(payload, ensure_ascii=False)


def build_rank_supervised_examples(
    samples: list[dict[str, Any]],
    *,
    prompt_path: str | Path,
    topk: int,
    include_reason: bool = False,
) -> list[RankSupervisedExample]:
    examples: list[RankSupervisedExample] = []
    prompt_path = Path(prompt_path)

    for sample in samples:
        candidate_item_ids = [str(item_id) for item_id in sample.get("candidate_item_ids", [])]
        prompt = build_candidate_ranking_prompt(sample, topk=topk, template_path=prompt_path)
        target_text = _build_target_text(sample, topk=topk, include_reason=include_reason)
        examples.append(
            RankSupervisedExample(
                source_event_id=str(sample.get("source_event_id", "")),
                user_id=str(sample.get("user_id", "")),
                prompt=prompt,
                target_text=target_text,
                positive_item_id=str(sample.get("positive_item_id", "")),
                candidate_item_ids=candidate_item_ids,
            )
        )
    return examples

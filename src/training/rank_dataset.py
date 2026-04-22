from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.llm.prompt_builder import build_candidate_ranking_prompt
from src.llm.prompt_builder import build_pairwise_preference_prompt
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


def summarize_rank_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "sample_count": 0,
            "avg_candidates": 0.0,
            "min_candidates": 0,
            "max_candidates": 0,
            "positive_in_candidate_rate": 0.0,
        }

    candidate_counts: list[int] = []
    positive_hits = 0
    for sample in samples:
        candidate_ids = [str(item_id) for item_id in sample.get("candidate_item_ids", [])]
        candidate_counts.append(len(candidate_ids))
        positive_item_id = str(sample.get("positive_item_id", "")).strip()
        if positive_item_id and positive_item_id in candidate_ids:
            positive_hits += 1

    return {
        "sample_count": len(samples),
        "avg_candidates": sum(candidate_counts) / len(candidate_counts),
        "min_candidates": min(candidate_counts),
        "max_candidates": max(candidate_counts),
        "positive_in_candidate_rate": positive_hits / len(samples),
    }


def _normalize_target_ranking(sample: dict[str, Any], topk: int) -> list[str]:
    candidate_ids = [str(item_id) for item_id in sample.get("candidate_item_ids", [])]
    positive_item_id = str(sample.get("positive_item_id", "")).strip()
    if not candidate_ids:
        raise ValueError("Ranking training sample is missing candidate_item_ids.")

    for teacher_key in (
        "srpd_teacher_ranked_item_ids",
        "teacher_ranked_item_ids",
        "target_ranked_item_ids",
        "pred_ranked_item_ids",
    ):
        teacher_order = sample.get(teacher_key)
        if isinstance(teacher_order, list):
            ordered = [str(item_id) for item_id in teacher_order if str(item_id) in candidate_ids]
            ordered.extend([item_id for item_id in candidate_ids if item_id not in ordered])
            if ordered:
                return ordered[:topk]

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


def _build_pair_sample(sample: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    pair_sample = dict(sample)
    pair_sample.update(
        {
            "item_a_id": str(pair.get("chosen_item_id", "")),
            "item_a_title": str(pair.get("chosen_item_title", "")),
            "item_a_text": "",
            "item_b_id": str(pair.get("rejected_item_id", "")),
            "item_b_title": str(pair.get("rejected_item_title", "")),
            "item_b_text": "",
        }
    )
    return pair_sample


def _build_preference_target_text(pair: dict[str, Any], *, include_reason: bool) -> str:
    chosen = str(pair.get("chosen_item_id", "")).strip()
    confidence = max(0.0, min(float(pair.get("preference_weight", 1.0)), 1.0))
    payload: dict[str, Any] = {
        "preferred_item": chosen,
        "confidence": confidence,
    }
    if include_reason:
        source = str(pair.get("preference_source", "structured_risk_preference")).replace("_", " ")
        payload["reason"] = f"The chosen item is preferred by the {source} signal."
    else:
        payload["reason"] = "The chosen item should rank ahead under the preference signal."
    return json.dumps(payload, ensure_ascii=False)


def build_rank_preference_examples(
    samples: list[dict[str, Any]],
    *,
    prompt_path: str | Path,
    include_reason: bool = False,
    max_pairs_per_sample: int | None = None,
) -> list[RankSupervisedExample]:
    examples: list[RankSupervisedExample] = []
    prompt_path = Path(prompt_path)

    for sample in samples:
        pairs = sample.get("srpd_dpo_style_preferences", [])
        if not isinstance(pairs, list) or not pairs:
            continue
        resolved_pairs = pairs[:max_pairs_per_sample] if max_pairs_per_sample is not None else pairs
        for pair_idx, pair in enumerate(resolved_pairs, start=1):
            if not isinstance(pair, dict):
                continue
            chosen = str(pair.get("chosen_item_id", "")).strip()
            rejected = str(pair.get("rejected_item_id", "")).strip()
            if not chosen or not rejected or chosen == rejected:
                continue
            pair_sample = _build_pair_sample(sample, pair)
            prompt = build_pairwise_preference_prompt(pair_sample, template_path=prompt_path)
            target_text = _build_preference_target_text(pair, include_reason=include_reason)
            examples.append(
                RankSupervisedExample(
                    source_event_id=f"{sample.get('source_event_id', '')}::pref::{pair_idx}",
                    user_id=str(sample.get("user_id", "")),
                    prompt=prompt,
                    target_text=target_text,
                    positive_item_id=str(sample.get("positive_item_id", "")),
                    candidate_item_ids=[chosen, rejected],
                )
            )
    return examples

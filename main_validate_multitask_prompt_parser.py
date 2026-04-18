from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.llm.parser import (
    parse_candidate_ranking_response,
    parse_pairwise_preference_response,
)
from src.llm.prompt_builder import PromptBuilder
from src.utils.io import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ranking_input",
        type=str,
        default="data/processed/amazon_beauty/ranking_valid.jsonl",
        help="Ranking sample input path.",
    )
    parser.add_argument(
        "--pairwise_input",
        type=str,
        default="data/processed/amazon_beauty/pairwise_valid.jsonl",
        help="Pairwise sample input path.",
    )
    parser.add_argument(
        "--ranking_prompt_path",
        type=str,
        default="prompts/candidate_ranking.txt",
        help="Ranking prompt template path.",
    )
    parser.add_argument(
        "--pairwise_prompt_path",
        type=str,
        default="prompts/pairwise_preference.txt",
        help="Pairwise prompt template path.",
    )
    parser.add_argument("--ranking_max_samples", type=int, default=12, help="Ranking validation sample cap.")
    parser.add_argument("--pairwise_max_samples", type=int, default=24, help="Pairwise validation sample cap.")
    parser.add_argument(
        "--summary_path",
        type=str,
        default="outputs/summary/week5_day2_parse_success.csv",
        help="Summary CSV output path.",
    )
    parser.add_argument(
        "--examples_path",
        type=str,
        default="outputs/summary/week5_day2_prompt_examples.md",
        help="Prompt example markdown output path.",
    )
    return parser.parse_args()


def _truncate(text: str, limit: int = 1200) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]..."


def _ranking_mock_response(sample: dict[str, Any], mode: str) -> str:
    item_ids = [str(item_id) for item_id in sample["candidate_item_ids"]]
    positive_item_id = str(sample["positive_item_id"])
    negatives = [item_id for item_id in item_ids if item_id != positive_item_id]
    ranking = [positive_item_id] + negatives
    topk = ranking[:3]

    if mode == "json_full":
        return json.dumps(
            {
                "ranked_item_ids": ranking,
                "topk_item_ids": topk,
                "confidence": 0.83,
                "reason": "history aligns most strongly with the first ranked candidate.",
            },
            ensure_ascii=False,
            indent=2,
        )
    if mode == "json_fenced":
        return (
            "```json\n"
            + json.dumps(
                {
                    "ranked_item_ids": ranking,
                    "topk_item_ids": topk,
                    "confidence": 83,
                    "reason": "the first candidate best matches the observed preference pattern.",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
            "```"
        )
    if mode == "topk_only":
        return json.dumps(
            {
                "topk_item_ids": topk,
                "confidence": "0.77",
                "reason": "these are the strongest items inside the candidate set.",
            },
            ensure_ascii=False,
            indent=2,
        )
    if mode == "invalid_ooc":
        return json.dumps(
            {
                "ranked_item_ids": ["OUT_OF_SET_ITEM", positive_item_id],
                "topk_item_ids": ["OUT_OF_SET_ITEM"],
                "confidence": 0.60,
                "reason": "the model drifted outside the candidate set.",
            },
            ensure_ascii=False,
            indent=2,
        )
    raise ValueError(f"Unsupported ranking mock mode: {mode}")


def _pairwise_mock_response(sample: dict[str, Any], mode: str) -> str:
    preferred_item = str(sample["preferred_item"])
    item_a_id = str(sample["item_a_id"])
    item_b_id = str(sample["item_b_id"])

    if mode == "json_item_id":
        return json.dumps(
            {
                "preferred_item": preferred_item,
                "confidence": 0.81,
                "reason": "this item better aligns with the recent history.",
            },
            ensure_ascii=False,
            indent=2,
        )
    if mode == "json_fenced_ab":
        preferred_symbol = "A" if preferred_item == item_a_id else "B"
        return (
            "```json\n"
            + json.dumps(
                {
                    "preferred_item": preferred_symbol,
                    "confidence": "81%",
                    "reason": "the chosen side is more compatible with the user trajectory.",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
            "```"
        )
    if mode == "natural_language":
        other_item = item_b_id if preferred_item == item_a_id else item_a_id
        return (
            f"I would choose {preferred_item} over {other_item} for this user. "
            "Confidence: 0.73. "
            "Reason: the selected candidate is the safer preference choice."
        )
    if mode == "ambiguous_invalid":
        return json.dumps(
            {
                "preferred_item": "maybe",
                "confidence": 0.50,
                "reason": "the answer is too ambiguous to be usable.",
            },
            ensure_ascii=False,
            indent=2,
        )
    raise ValueError(f"Unsupported pairwise mock mode: {mode}")


def _build_markdown_examples(examples: list[dict[str, Any]]) -> str:
    lines = ["# Week5 Day2 Prompt and Parser Examples", ""]
    for example in examples:
        lines.append(f"## {example['task']} | {example['mode']} | success={example['parse_success']}")
        lines.append("")
        lines.append("Prompt:")
        lines.append("```text")
        lines.append(_truncate(example["prompt"]))
        lines.append("```")
        lines.append("")
        lines.append("Response:")
        lines.append("```text")
        lines.append(_truncate(example["response"]))
        lines.append("```")
        lines.append("")
        lines.append("Parsed:")
        lines.append("```json")
        lines.append(_truncate(example["parsed"]))
        lines.append("```")
        lines.append("")
        lines.append(f"Repair note: {example['note']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    ranking_samples = load_jsonl(args.ranking_input)[: args.ranking_max_samples]
    pairwise_samples = load_jsonl(args.pairwise_input)[: args.pairwise_max_samples]

    ranking_builder = PromptBuilder(args.ranking_prompt_path)
    pairwise_builder = PromptBuilder(args.pairwise_prompt_path)

    ranking_modes = ["json_full"] * max(0, len(ranking_samples) - 2) + ["json_fenced", "invalid_ooc"]
    ranking_modes = ranking_modes[: len(ranking_samples)]
    if len(ranking_modes) >= 2:
        ranking_modes[-2] = "topk_only"
        ranking_modes[-1] = "invalid_ooc"

    pairwise_modes = ["json_item_id"] * max(0, len(pairwise_samples) - 3) + ["json_fenced_ab", "natural_language", "ambiguous_invalid"]
    pairwise_modes = pairwise_modes[: len(pairwise_samples)]
    if len(pairwise_modes) >= 3:
        pairwise_modes[-3] = "json_fenced_ab"
        pairwise_modes[-2] = "natural_language"
        pairwise_modes[-1] = "ambiguous_invalid"

    summary_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    captured_example_modes: set[tuple[str, str]] = set()

    ranking_output_lengths: list[int] = []
    ranking_parse_success_values: list[int] = []
    ranking_illegal_values: list[int] = []
    ranking_ooc_values: list[int] = []

    for sample, mode in zip(ranking_samples, ranking_modes):
        prompt = ranking_builder.build_candidate_ranking_prompt(sample, topk=3)
        response = _ranking_mock_response(sample, mode)
        parsed = parse_candidate_ranking_response(
            response,
            allowed_item_ids=[str(item_id) for item_id in sample["candidate_item_ids"]],
            topk=3,
        )
        ranking_output_lengths.append(len(response))
        ranking_parse_success_values.append(int(parsed["parse_success"]))
        ranking_illegal_values.append(int(not parsed["parse_success"]))
        ranking_ooc_values.append(int(parsed["contains_out_of_candidate_item"]))

        if (("candidate_ranking", mode) not in captured_example_modes) and mode in {
            "json_full",
            "json_fenced",
            "topk_only",
            "invalid_ooc",
        }:
            captured_example_modes.add(("candidate_ranking", mode))
            example_rows.append(
                {
                    "task": "candidate_ranking",
                    "mode": mode,
                    "parse_success": parsed["parse_success"],
                    "prompt": prompt,
                    "response": response,
                    "parsed": json_dumps(parsed),
                    "note": "Ranking parser should accept strict JSON, fenced JSON, and top-k-only outputs while explicitly rejecting out-of-candidate drift.",
                }
            )

    summary_rows.append(
        {
            "task": "candidate_ranking",
            "sample_count": len(ranking_samples),
            "parse_success_rate": round(mean(ranking_parse_success_values), 4) if ranking_parse_success_values else 0.0,
            "avg_output_length": round(mean(ranking_output_lengths), 2) if ranking_output_lengths else 0.0,
            "illegal_output_ratio": round(mean(ranking_illegal_values), 4) if ranking_illegal_values else 0.0,
            "out_of_candidate_item_ratio": round(mean(ranking_ooc_values), 4) if ranking_ooc_values else 0.0,
            "ambiguous_preference_ratio": 0.0,
            "notes": "Parser self-check over real ranking samples with strict/fenced/partial JSON plus out-of-candidate stress cases.",
        }
    )

    pairwise_output_lengths: list[int] = []
    pairwise_parse_success_values: list[int] = []
    pairwise_illegal_values: list[int] = []
    pairwise_ambiguous_values: list[int] = []

    for sample, mode in zip(pairwise_samples, pairwise_modes):
        prompt = pairwise_builder.build_pairwise_preference_prompt(sample)
        response = _pairwise_mock_response(sample, mode)
        parsed = parse_pairwise_preference_response(
            response,
            item_a_id=str(sample["item_a_id"]),
            item_b_id=str(sample["item_b_id"]),
        )
        pairwise_output_lengths.append(len(response))
        pairwise_parse_success_values.append(int(parsed["parse_success"]))
        pairwise_illegal_values.append(int(not parsed["parse_success"]))
        pairwise_ambiguous_values.append(int(parsed["ambiguous_preference"]))

        if (("pairwise_preference", mode) not in captured_example_modes) and mode in {
            "json_item_id",
            "json_fenced_ab",
            "natural_language",
            "ambiguous_invalid",
        }:
            captured_example_modes.add(("pairwise_preference", mode))
            example_rows.append(
                {
                    "task": "pairwise_preference",
                    "mode": mode,
                    "parse_success": parsed["parse_success"],
                    "prompt": prompt,
                    "response": response,
                    "parsed": json_dumps(parsed),
                    "note": "Pairwise parser should handle explicit item ids, fenced A/B shorthand, and free-form preference text while preserving an explicit ambiguous failure case.",
                }
            )

    summary_rows.append(
        {
            "task": "pairwise_preference",
            "sample_count": len(pairwise_samples),
            "parse_success_rate": round(mean(pairwise_parse_success_values), 4) if pairwise_parse_success_values else 0.0,
            "avg_output_length": round(mean(pairwise_output_lengths), 2) if pairwise_output_lengths else 0.0,
            "illegal_output_ratio": round(mean(pairwise_illegal_values), 4) if pairwise_illegal_values else 0.0,
            "out_of_candidate_item_ratio": 0.0,
            "ambiguous_preference_ratio": round(mean(pairwise_ambiguous_values), 4) if pairwise_ambiguous_values else 0.0,
            "notes": "Parser self-check over real pairwise samples with item-id JSON, A/B shorthand, natural language wrapping, and ambiguous failures.",
        }
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    examples_md = _build_markdown_examples(example_rows)
    examples_path = Path(args.examples_path)
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    examples_path.write_text(examples_md, encoding="utf-8")

    print(f"[Saved] {summary_path}")
    print(f"[Saved] {examples_path}")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

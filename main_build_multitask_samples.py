from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from src.data.candidate_ranking_builder import (
    CandidateRankingBuildConfig,
    build_candidate_ranking_samples_from_pointwise,
)
from src.data.pairwise_preference_builder import (
    PairwisePreferenceBuildConfig,
    build_pairwise_preferences_from_ranking_samples,
)
from src.utils.io import load_jsonl, save_jsonl


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the data config file.")
    parser.add_argument(
        "--ranking_task_config",
        type=str,
        default="configs/task/candidate_ranking.yaml",
        help="Path to the candidate ranking task config.",
    )
    parser.add_argument(
        "--pairwise_task_config",
        type=str,
        default="configs/task/pairwise_preference.yaml",
        help="Path to the pairwise preference task config.",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="outputs/summary/week5_day1_multitask_data_summary.csv",
        help="Where to save the multitask data summary table.",
    )
    return parser.parse_args()


def validate_ranking_samples(samples: List[Dict[str, Any]], split_name: str) -> None:
    if not samples:
        raise ValueError(f"[{split_name}] No candidate ranking samples were generated.")

    for idx, sample in enumerate(samples):
        candidate_item_ids = sample.get("candidate_item_ids", [])
        positive_item_id = str(sample.get("positive_item_id", "")).strip()
        history = sample.get("history", [])

        if not isinstance(history, list):
            raise ValueError(f"[{split_name}] Ranking sample {idx} has non-list history.")
        if not candidate_item_ids:
            raise ValueError(f"[{split_name}] Ranking sample {idx} has empty candidate_item_ids.")
        if positive_item_id == "":
            raise ValueError(f"[{split_name}] Ranking sample {idx} has empty positive_item_id.")
        if positive_item_id not in candidate_item_ids:
            raise ValueError(f"[{split_name}] Ranking sample {idx} does not include the positive item.")


def validate_pairwise_samples(
    pairwise_samples: List[Dict[str, Any]],
    ranking_samples: List[Dict[str, Any]],
    split_name: str,
) -> None:
    if not pairwise_samples:
        raise ValueError(f"[{split_name}] No pairwise preference samples were generated.")

    ranking_event_ids = {str(sample["source_event_id"]) for sample in ranking_samples}
    pairwise_event_ids = set()

    for idx, sample in enumerate(pairwise_samples):
        preferred_item = str(sample.get("preferred_item", "")).strip()
        item_a_id = str(sample.get("item_a_id", "")).strip()
        item_b_id = str(sample.get("item_b_id", "")).strip()
        source_event_id = str(sample.get("source_event_id", "")).strip()

        if preferred_item == "":
            raise ValueError(f"[{split_name}] Pairwise sample {idx} has empty preferred_item.")
        if item_a_id == "" or item_b_id == "":
            raise ValueError(f"[{split_name}] Pairwise sample {idx} is missing item ids.")
        if preferred_item not in {item_a_id, item_b_id}:
            raise ValueError(f"[{split_name}] Pairwise sample {idx} preferred_item is not in the pair.")
        if source_event_id not in ranking_event_ids:
            raise ValueError(f"[{split_name}] Pairwise sample {idx} has unknown source_event_id.")

        pairwise_event_ids.add(source_event_id)

    if not pairwise_event_ids:
        raise ValueError(f"[{split_name}] No valid pairwise source events were found.")


def build_split_summary(
    split_name: str,
    ranking_samples: List[Dict[str, Any]],
    pairwise_samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    ranking_count = len(ranking_samples)
    pairwise_count = len(pairwise_samples)

    avg_candidate_count = (
        sum(int(sample["num_candidates"]) for sample in ranking_samples) / ranking_count
        if ranking_count > 0
        else 0.0
    )
    avg_pair_count = pairwise_count / ranking_count if ranking_count > 0 else 0.0

    pair_type_counts: Dict[str, int] = {}
    pairwise_event_ids = set()
    for sample in pairwise_samples:
        pair_type = str(sample.get("pair_type", "unknown"))
        pair_type_counts[pair_type] = pair_type_counts.get(pair_type, 0) + 1
        pairwise_event_ids.add(str(sample["source_event_id"]))

    positive_coverage_rate = (
        sum(
            1
            for sample in ranking_samples
            if str(sample["positive_item_id"]) in {str(item_id) for item_id in sample["candidate_item_ids"]}
        )
        / ranking_count
        if ranking_count > 0
        else 0.0
    )
    pairwise_source_coverage_rate = len(pairwise_event_ids) / ranking_count if ranking_count > 0 else 0.0

    return {
        "split_name": split_name,
        "ranking_sample_count": ranking_count,
        "pairwise_sample_count": pairwise_count,
        "avg_candidate_count": round(avg_candidate_count, 4),
        "avg_pair_count": round(avg_pair_count, 4),
        "pair_type_distribution": json.dumps(pair_type_counts, ensure_ascii=False, sort_keys=True),
        "positive_coverage_rate": round(positive_coverage_rate, 4),
        "pairwise_source_coverage_rate": round(pairwise_source_coverage_rate, 4),
    }


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.config)
    ranking_task_cfg = load_yaml(args.ranking_task_config)
    pairwise_task_cfg = load_yaml(args.pairwise_task_config)

    processed_dir = Path(data_cfg["processed_dir"])
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    ranking_builder_cfg = ranking_task_cfg.get("builder", {})
    pairwise_builder_cfg = pairwise_task_cfg.get("builder", {})

    source_splits = ranking_builder_cfg.get("source_splits", ["valid", "test"])
    if source_splits != pairwise_builder_cfg.get("source_splits", source_splits):
        raise ValueError("Ranking and pairwise task configs must use the same source_splits in week5 day1.")

    ranking_cfg = CandidateRankingBuildConfig(
        shuffle_candidates=bool(ranking_builder_cfg.get("shuffle_candidates", True)),
        shuffle_seed=int(ranking_builder_cfg.get("shuffle_seed", 42)),
        require_single_positive=bool(ranking_builder_cfg.get("require_single_positive", True)),
    )
    pairwise_cfg = PairwisePreferenceBuildConfig(
        pair_type=str(pairwise_builder_cfg.get("pair_type", "positive_vs_negative")),
        pair_generation_mode=str(pairwise_builder_cfg.get("pair_generation_mode", "positive_vs_negative")),
        shuffle_pair_order=bool(pairwise_builder_cfg.get("shuffle_pair_order", True)),
        shuffle_seed=int(pairwise_builder_cfg.get("shuffle_seed", 42)),
        max_pairs_per_sample=pairwise_builder_cfg.get("max_pairs_per_sample"),
        max_pairs_per_event=pairwise_builder_cfg.get("max_pairs_per_event"),
        event_balanced_order=bool(pairwise_builder_cfg.get("event_balanced_order", False)),
    )

    ranking_prefix = str(ranking_builder_cfg.get("output_prefix", "ranking")).strip() or "ranking"
    pairwise_prefix = str(pairwise_builder_cfg.get("output_prefix", "pairwise")).strip() or "pairwise"

    summary_rows: List[Dict[str, Any]] = []

    for split_name in source_splits:
        pointwise_path = processed_dir / f"{split_name}.jsonl"
        if not pointwise_path.exists():
            raise FileNotFoundError(f"Pointwise split not found: {pointwise_path}")

        pointwise_records = load_jsonl(pointwise_path)
        ranking_samples = build_candidate_ranking_samples_from_pointwise(
            pointwise_records,
            split_name=split_name,
            cfg=ranking_cfg,
        )
        validate_ranking_samples(ranking_samples, split_name)

        pairwise_samples = build_pairwise_preferences_from_ranking_samples(
            ranking_samples,
            cfg=pairwise_cfg,
        )
        validate_pairwise_samples(pairwise_samples, ranking_samples, split_name)

        ranking_output_path = processed_dir / f"{ranking_prefix}_{split_name}.jsonl"
        pairwise_output_path = processed_dir / f"{pairwise_prefix}_{split_name}.jsonl"
        save_jsonl(ranking_samples, ranking_output_path)
        save_jsonl(pairwise_samples, pairwise_output_path)

        print(f"[Saved] {ranking_output_path} ({len(ranking_samples)} samples)")
        print(f"[Saved] {pairwise_output_path} ({len(pairwise_samples)} samples)")

        summary_rows.append(build_split_summary(split_name, ranking_samples, pairwise_samples))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()

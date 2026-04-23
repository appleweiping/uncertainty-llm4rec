from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.candidate_ranking_builder import (
    CandidateRankingBuildConfig,
    build_candidate_ranking_samples_from_pointwise,
)
from src.data.sample_builder import (
    BuildSamplesConfig,
    build_eval_samples_for_split,
    build_item_lookup,
    build_popularity_lookup,
    build_train_pointwise_samples,
    deduplicate_user_sequences,
    sort_and_group_interactions,
    split_user_sequence_leave_one_out,
)
from src.utils.io import load_jsonl, save_jsonl


DEFAULT_CONFIGS = [
    "configs/data/amazon_beauty.yaml",
    "configs/data/amazon_books.yaml",
    "configs/data/amazon_electronics.yaml",
    "configs/data/amazon_movies.yaml",
]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize full-domain pointwise and ranking runtime files for Week7.8 local-v2 replay."
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="Data config paths to materialize.",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="outputs/summary/week7_8_replay_v2_runtime_materialization.csv",
        help="Where to save the runtime materialization summary.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild pointwise and ranking runtime files even if they already exist.",
    )
    return parser.parse_args()


def _required_csv_paths(processed_dir: Path) -> dict[str, Path]:
    return {
        "interactions": processed_dir / "interactions.csv",
        "items": processed_dir / "items.csv",
        "popularity": processed_dir / "popularity_stats.csv",
    }


def _build_pointwise_splits(processed_dir: Path, cfg: dict[str, Any], overwrite: bool) -> dict[str, int]:
    output_paths = {
        "train": processed_dir / "train.jsonl",
        "valid": processed_dir / "valid.jsonl",
        "test": processed_dir / "test.jsonl",
    }
    if not overwrite and all(path.exists() for path in output_paths.values()):
        return {f"{split}_rows": len(load_jsonl(path)) for split, path in output_paths.items()}

    csv_paths = _required_csv_paths(processed_dir)
    for label, path in csv_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required processed file for {label}: {path}")

    interactions_df = pd.read_csv(csv_paths["interactions"])
    items_df = pd.read_csv(csv_paths["items"])
    popularity_df = pd.read_csv(csv_paths["popularity"])

    sample_cfg = BuildSamplesConfig(
        max_history_len=cfg.get("sampling", {}).get("max_history_len", 10),
        num_negatives=cfg.get("sampling", {}).get("num_negatives", 5),
        seed=cfg.get("sampling", {}).get("seed", 42),
    )
    min_sequence_length = cfg.get("split", {}).get("min_sequence_length", 3)

    item_lookup = build_item_lookup(items_df)
    popularity_lookup = build_popularity_lookup(popularity_df)
    user_sequences = deduplicate_user_sequences(sort_and_group_interactions(interactions_df))
    user_sequences = {
        user_id: seq
        for user_id, seq in user_sequences.items()
        if len(seq) >= min_sequence_length
    }
    train_histories, valid_targets, test_targets = split_user_sequence_leave_one_out(user_sequences)

    train_records = build_train_pointwise_samples(
        train_histories=train_histories,
        item_lookup=item_lookup,
        popularity_lookup=popularity_lookup,
        cfg=sample_cfg,
    )
    valid_records = build_eval_samples_for_split(
        train_histories=train_histories,
        split_targets=valid_targets,
        item_lookup=item_lookup,
        popularity_lookup=popularity_lookup,
        cfg=sample_cfg,
    )
    test_records = build_eval_samples_for_split(
        train_histories=train_histories,
        split_targets=test_targets,
        item_lookup=item_lookup,
        popularity_lookup=popularity_lookup,
        cfg=sample_cfg,
    )

    save_jsonl(train_records, output_paths["train"])
    save_jsonl(valid_records, output_paths["valid"])
    save_jsonl(test_records, output_paths["test"])

    return {
        "train_rows": len(train_records),
        "valid_rows": len(valid_records),
        "test_rows": len(test_records),
    }


def _build_ranking_splits(processed_dir: Path, seed: int, overwrite: bool) -> dict[str, int]:
    output_paths = {
        "ranking_valid": processed_dir / "ranking_valid.jsonl",
        "ranking_test": processed_dir / "ranking_test.jsonl",
    }
    if not overwrite and all(path.exists() for path in output_paths.values()):
        return {f"{split}_rows": len(load_jsonl(path)) for split, path in output_paths.items()}

    build_cfg = CandidateRankingBuildConfig(
        shuffle_candidates=True,
        shuffle_seed=seed,
        require_single_positive=True,
    )

    valid_records = load_jsonl(processed_dir / "valid.jsonl")
    test_records = load_jsonl(processed_dir / "test.jsonl")

    ranking_valid = build_candidate_ranking_samples_from_pointwise(
        valid_records,
        split_name="valid",
        cfg=build_cfg,
    )
    ranking_test = build_candidate_ranking_samples_from_pointwise(
        test_records,
        split_name="test",
        cfg=build_cfg,
    )

    save_jsonl(ranking_valid, output_paths["ranking_valid"])
    save_jsonl(ranking_test, output_paths["ranking_test"])

    return {
        "ranking_valid_rows": len(ranking_valid),
        "ranking_test_rows": len(ranking_test),
    }


def _build_domain_runtime(config_path: str | Path, overwrite: bool) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    processed_dir = Path(cfg["processed_dir"])
    row: dict[str, Any] = {
        "config_path": str(config_path),
        "processed_dir": str(processed_dir),
        "dataset_name": cfg.get("dataset_name", ""),
        "domain_name": cfg.get("domain_name", ""),
        "status": "ok",
        "notes": "",
    }

    try:
        pointwise_counts = _build_pointwise_splits(processed_dir, cfg, overwrite)
        ranking_counts = _build_ranking_splits(
            processed_dir,
            seed=cfg.get("sampling", {}).get("seed", 42),
            overwrite=overwrite,
        )
        row.update(pointwise_counts)
        row.update(ranking_counts)
    except Exception as exc:  # noqa: BLE001 - explicit summary capture for runtime alignment
        row["status"] = "failed"
        row["notes"] = str(exc)
        for col in [
            "train_rows",
            "valid_rows",
            "test_rows",
            "ranking_valid_rows",
            "ranking_test_rows",
        ]:
            row.setdefault(col, pd.NA)

    return row


def main() -> None:
    args = parse_args()
    rows = [_build_domain_runtime(config_path, args.overwrite) for config_path in args.configs]
    summary_df = pd.DataFrame(rows)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved Week7.8 replay runtime materialization summary to: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

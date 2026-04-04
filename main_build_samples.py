from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

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


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_jsonl(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_records(records, split_name: str) -> None:
    total = len(records)
    pos_cnt = sum(1 for r in records if int(r.get("label", 0)) == 1)
    neg_cnt = sum(1 for r in records if int(r.get("label", 0)) == 0)

    if total == 0:
        raise ValueError(f"[{split_name}] 样本数为 0，构造失败")
    if pos_cnt == 0:
        raise ValueError(f"[{split_name}] 没有任何正样本，构造失败")
    if neg_cnt == 0:
        raise ValueError(f"[{split_name}] 没有任何负样本，构造失败")

    for i, rec in enumerate(records):
        candidate_item_id = str(rec.get("candidate_item_id", "")).strip()
        candidate_text = str(rec.get("candidate_text", "")).strip()
        history = rec.get("history", [])

        if candidate_item_id == "":
            raise ValueError(f"[{split_name}] 第 {i} 条样本缺少 candidate_item_id")
        if candidate_text == "":
            raise ValueError(f"[{split_name}] 第 {i} 条样本缺少 candidate_text")
        if not isinstance(history, list):
            raise ValueError(f"[{split_name}] 第 {i} 条样本 history 不是 list")

    print(f"[Validate] {split_name}: total={total}, positive={pos_cnt}, negative={neg_cnt}, passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径，例如 configs/data/amazon_beauty.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    processed_dir = Path(cfg["processed_dir"])
    interactions_path = processed_dir / "interactions.csv"
    items_path = processed_dir / "items.csv"
    popularity_path = processed_dir / "popularity_stats.csv"

    if not interactions_path.exists():
        raise FileNotFoundError(f"未找到文件: {interactions_path}")
    if not items_path.exists():
        raise FileNotFoundError(f"未找到文件: {items_path}")
    if not popularity_path.exists():
        raise FileNotFoundError(f"未找到文件: {popularity_path}")

    interactions_df = pd.read_csv(interactions_path)
    items_df = pd.read_csv(items_path)
    popularity_df = pd.read_csv(popularity_path)

    sample_cfg = BuildSamplesConfig(
        max_history_len=cfg.get("sampling", {}).get("max_history_len", 10),
        num_negatives=cfg.get("sampling", {}).get("num_negatives", 5),
        seed=cfg.get("sampling", {}).get("seed", 42),
    )

    split_method = cfg.get("split", {}).get("method", "leave_one_out")
    min_sequence_length = cfg.get("split", {}).get("min_sequence_length", 3)

    if split_method != "leave_one_out":
        raise ValueError(f"当前仅支持 split.method = leave_one_out，收到: {split_method}")

    item_lookup = build_item_lookup(items_df)
    popularity_lookup = build_popularity_lookup(popularity_df)

    user_sequences = sort_and_group_interactions(interactions_df)
    print(f"[Info] raw users: {len(user_sequences)}")

    user_sequences = deduplicate_user_sequences(user_sequences)
    print(f"[Info] deduplicated users: {len(user_sequences)}")

    user_sequences = {
        user_id: seq
        for user_id, seq in user_sequences.items()
        if len(seq) >= min_sequence_length
    }
    print(f"[Info] users after min_sequence_length filter: {len(user_sequences)}")

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

    validate_records(train_records, "train")
    validate_records(valid_records, "valid")
    validate_records(test_records, "test")

    save_jsonl(train_records, processed_dir / "train.jsonl")
    save_jsonl(valid_records, processed_dir / "valid.jsonl")
    save_jsonl(test_records, processed_dir / "test.jsonl")

    print(f"[Done] train samples: {len(train_records)}")
    print(f"[Done] valid samples: {len(valid_records)}")
    print(f"[Done] test samples: {len(test_records)}")
    print(f"[Saved] {processed_dir / 'train.jsonl'}")
    print(f"[Saved] {processed_dir / 'valid.jsonl'}")
    print(f"[Saved] {processed_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()
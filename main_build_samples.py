from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml


def load_config(config_path: str | Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_jsonl(records: List[Dict], path: str | Path) -> None:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_history_records(
    history_item_ids: List[int],
    item_meta: Dict[int, Dict],
    max_history_length: int,
) -> Tuple[List[Dict], List[str]]:
    truncated = history_item_ids[-max_history_length:]
    history = []
    history_text = []

    for item_id in truncated:
        meta = item_meta[item_id]
        entry = {
            "item_id": int(item_id),
            "title": meta["title"],
            "text": meta["item_text"],
        }
        history.append(entry)
        history_text.append(f"{meta['title']} | {meta['item_text']}")

    return history, history_text


def sample_negative_items(
    candidate_pool: List[int],
    num_negatives: int,
    rng: random.Random,
) -> List[int]:
    if len(candidate_pool) <= num_negatives:
        return candidate_pool.copy()
    return rng.sample(candidate_pool, num_negatives)


def split_user_positive_positions(
    num_positive_instances: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> Tuple[List[int], List[int], List[int]]:
    """
    将一个用户可用的正样本位置切成 train/valid/test。
    """
    if num_positive_instances < 3:
        raise ValueError("Each user must have at least 3 positive instances after min_history filtering.")

    total = num_positive_instances
    n_train = max(1, int(total * train_ratio))
    n_valid = max(1, int(total * valid_ratio))
    n_test = max(1, total - n_train - n_valid)

    while n_train + n_valid + n_test > total:
        if n_train >= n_valid and n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_valid >= n_test and n_valid > 1:
            n_valid -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break

    while n_train + n_valid + n_test < total:
        n_train += 1

    indices = list(range(total))
    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:n_train + n_valid + n_test]
    return train_idx, valid_idx, test_idx


def build_samples(config: Dict) -> None:
    seed = int(config["seed"])
    rng = random.Random(seed)

    processed_dir = Path(config["processed_dir"])
    interactions_path = processed_dir / "interactions.csv"
    items_path = processed_dir / "items.csv"
    popularity_path = processed_dir / "popularity_stats.csv"

    if not interactions_path.exists():
        raise FileNotFoundError(f"Missing file: {interactions_path}")
    if not items_path.exists():
        raise FileNotFoundError(f"Missing file: {items_path}")
    if not popularity_path.exists():
        raise FileNotFoundError(f"Missing file: {popularity_path}")

    interactions_df = pd.read_csv(interactions_path)
    items_df = pd.read_csv(items_path)
    popularity_df = pd.read_csv(popularity_path)

    min_history_length = int(config["sample_builder"]["min_history_length"])
    max_history_length = int(config["sample_builder"]["max_history_length"])
    negatives_per_positive = int(config["sample_builder"]["negatives_per_positive"])

    train_ratio = float(config["sample_builder"]["train_ratio"])
    valid_ratio = float(config["sample_builder"]["valid_ratio"])
    test_ratio = float(config["sample_builder"]["test_ratio"])

    item_meta = {}
    for _, row in items_df.iterrows():
        item_meta[int(row["item_id"])] = {
            "title": str(row["title"]),
            "item_text": str(row["item_text"]),
            "genres": str(row.get("genres", "")),
            "popularity_group": str(row.get("popularity_group", "")),
        }

    popularity_map = {
        int(row["item_id"]): str(row["popularity_group"])
        for _, row in popularity_df.iterrows()
    }

    all_item_ids = sorted(item_meta.keys())

    interactions_df = interactions_df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    user_groups = interactions_df.groupby("user_id")

    train_records: List[Dict] = []
    valid_records: List[Dict] = []
    test_records: List[Dict] = []

    skipped_users = 0
    total_positive_instances = 0

    for user_id, group in user_groups:
        seq = group["item_id"].astype(int).tolist()

        # 至少要有 min_history_length 作为历史，再加至少 3 个可预测正样本，才能切 train/valid/test
        if len(seq) < min_history_length + 3:
            skipped_users += 1
            continue

        candidate_positions = list(range(min_history_length, len(seq)))
        train_idx, valid_idx, test_idx = split_user_positive_positions(
            num_positive_instances=len(candidate_positions),
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
        )

        user_all_interacted = set(seq)

        def add_record(target_list: List[Dict], pos_in_candidate_list: int) -> None:
            target_pos = candidate_positions[pos_in_candidate_list]
            history_item_ids = seq[:target_pos]
            positive_item_id = int(seq[target_pos])

            history, history_text = build_history_records(
                history_item_ids=history_item_ids,
                item_meta=item_meta,
                max_history_length=max_history_length,
            )

            # 正样本
            positive_meta = item_meta[positive_item_id]
            target_list.append(
                {
                    "user_id": str(user_id),
                    "target_item_id": str(positive_item_id),
                    "candidate_item_id": str(positive_item_id),
                    "candidate_title": positive_meta["title"],
                    "candidate_text": positive_meta["item_text"],
                    "label": 1,
                    "target_popularity_group": popularity_map.get(positive_item_id, "unknown"),
                    "history": history,
                    "history_text": history_text,
                }
            )

            # 负样本
            negative_pool = [iid for iid in all_item_ids if iid not in user_all_interacted]
            negative_ids = sample_negative_items(
                candidate_pool=negative_pool,
                num_negatives=negatives_per_positive,
                rng=rng,
            )

            for neg_item_id in negative_ids:
                neg_meta = item_meta[int(neg_item_id)]
                target_list.append(
                    {
                        "user_id": str(user_id),
                        "target_item_id": str(positive_item_id),
                        "candidate_item_id": str(neg_item_id),
                        "candidate_title": neg_meta["title"],
                        "candidate_text": neg_meta["item_text"],
                        "label": 0,
                        "target_popularity_group": popularity_map.get(positive_item_id, "unknown"),
                        "history": history,
                        "history_text": history_text,
                    }
                )

        for idx in train_idx:
            add_record(train_records, idx)
            total_positive_instances += 1
        for idx in valid_idx:
            add_record(valid_records, idx)
            total_positive_instances += 1
        for idx in test_idx:
            add_record(test_records, idx)
            total_positive_instances += 1

    train_out = processed_dir / "train.jsonl"
    valid_out = processed_dir / "valid.jsonl"
    test_out = processed_dir / "test.jsonl"

    save_jsonl(train_records, train_out)
    save_jsonl(valid_records, valid_out)
    save_jsonl(test_records, test_out)

    print(f"Saved train samples to: {train_out} ({len(train_records)} rows)")
    print(f"Saved valid samples to: {valid_out} ({len(valid_records)} rows)")
    print(f"Saved test samples to: {test_out} ({len(test_records)} rows)")
    print(f"Total positive instances used: {total_positive_instances}")
    print(f"Skipped users due to insufficient sequence length: {skipped_users}")
    print("Sample building completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pointwise train/valid/test jsonl samples.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data/movielens_1m.yaml",
        help="Path to data config yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset_name = config.get("dataset_name", "")
    if dataset_name != "movielens_1m":
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    build_samples(config)


if __name__ == "__main__":
    main()
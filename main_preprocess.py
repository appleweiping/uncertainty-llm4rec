from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml

from src.data.raw_loaders import load_movielens_1m
from src.data.text_builder import build_movielens_item_text


def load_config(config_path: str | Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def filter_interactions(
    ratings_df: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
    positive_rating_threshold: float,
) -> pd.DataFrame:
    """
    仅保留正反馈样本，并进行用户/物品频次过滤。
    """
    df = ratings_df.copy()
    df = df[df["rating"] >= positive_rating_threshold].copy()

    changed = True
    while changed:
        prev_len = len(df)

        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df["user_id"].isin(valid_users)].copy()

        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df["item_id"].isin(valid_items)].copy()

        changed = len(df) != prev_len

    return df


def build_popularity_stats(
    interactions_df: pd.DataFrame,
    bucket_names: Tuple[str, str, str] = ("tail", "mid", "head"),
) -> pd.DataFrame:
    """
    基于 item 交互次数构造 popularity group。
    """
    item_counts = (
        interactions_df.groupby("item_id")
        .size()
        .reset_index(name="interaction_count")
        .sort_values(["interaction_count", "item_id"], ascending=[True, True])
        .reset_index(drop=True)
    )

    n_items = len(item_counts)
    if n_items == 0:
        raise ValueError("No items found when building popularity stats.")

    # 按排序后位置均分成三段
    boundaries = [0, n_items // 3, (2 * n_items) // 3, n_items]

    groups = []
    for idx in range(n_items):
        if boundaries[0] <= idx < boundaries[1]:
            groups.append(bucket_names[0])
        elif boundaries[1] <= idx < boundaries[2]:
            groups.append(bucket_names[1])
        else:
            groups.append(bucket_names[2])

    item_counts["popularity_group"] = groups
    return item_counts


def preprocess_movielens(config: Dict) -> None:
    raw_dir = config["raw_dir"]
    processed_dir = ensure_dir(config["processed_dir"])

    min_user_interactions = int(config["min_user_interactions"])
    min_item_interactions = int(config["min_item_interactions"])
    positive_rating_threshold = float(config["labeling"]["positive_rating_threshold"])

    bucket_names = tuple(config["popularity"]["bucket_names"])

    print(f"Loading raw MovieLens 1M data from: {raw_dir}")
    ratings_df, movies_df, users_df = load_movielens_1m(raw_dir)

    print("Filtering interactions...")
    interactions_df = filter_interactions(
        ratings_df=ratings_df,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
        positive_rating_threshold=positive_rating_threshold,
    )

    valid_item_ids = set(interactions_df["item_id"].unique())
    valid_user_ids = set(interactions_df["user_id"].unique())

    movies_df = movies_df[movies_df["item_id"].isin(valid_item_ids)].copy()
    users_df = users_df[users_df["user_id"].isin(valid_user_ids)].copy()

    print("Building item text...")
    movies_df = build_movielens_item_text(
        movies_df=movies_df,
        use_title=bool(config["text_builder"]["use_title"]),
        use_genres=bool(config["text_builder"]["use_genres"]),
        genre_sep=str(config["text_builder"]["genre_sep"]),
    )

    print("Building popularity stats...")
    popularity_df = build_popularity_stats(
        interactions_df=interactions_df,
        bucket_names=bucket_names,
    )

    movies_df = movies_df.merge(
        popularity_df[["item_id", "interaction_count", "popularity_group"]],
        on="item_id",
        how="left",
    )

    interactions_out = processed_dir / "interactions.csv"
    items_out = processed_dir / "items.csv"
    users_out = processed_dir / "users.csv"
    popularity_out = processed_dir / "popularity_stats.csv"

    interactions_df = interactions_df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    movies_df = movies_df.sort_values("item_id").reset_index(drop=True)
    users_df = users_df.sort_values("user_id").reset_index(drop=True)
    popularity_df = popularity_df.sort_values("item_id").reset_index(drop=True)

    interactions_df.to_csv(interactions_out, index=False)
    movies_df.to_csv(items_out, index=False)
    users_df.to_csv(users_out, index=False)
    popularity_df.to_csv(popularity_out, index=False)

    print(f"Saved interactions to: {interactions_out}")
    print(f"Saved items to: {items_out}")
    print(f"Saved users to: {users_out}")
    print(f"Saved popularity stats to: {popularity_out}")
    print("Preprocessing completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MovieLens 1M into processed tables.")
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

    preprocess_movielens(config)


if __name__ == "__main__":
    main()
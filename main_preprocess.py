# main_preprocess.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.data.popularity import compute_item_popularity, build_popularity_groups
from src.data.raw_loaders import load_amazon_domain
from src.data.text_builder import attach_candidate_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径，例如 configs/data/amazon_beauty.yaml")
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    processed_dir = Path(cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    interactions, items, users, stats = load_amazon_domain(
        domain_name=cfg["domain_name"],
        review_path=cfg["raw"]["review_path"],
        meta_path=cfg["raw"]["meta_path"],
        rating_threshold=cfg["filter"]["rating_threshold"],
        min_user_interactions=cfg["filter"]["min_user_interactions"],
        min_item_interactions=cfg["filter"]["min_item_interactions"],
    )

    items = attach_candidate_text(
        items,
        strategy=cfg["text"]["strategy"],
        max_desc_len=cfg["text"].get("max_desc_len", 1000),
        fallback_to_title_categories=cfg["text"].get("fallback_to_title_categories", True),
    )

    interactions = interactions.copy()
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["item_id"].astype(str)

    grouped = (
        interactions.sort_values(["user_id", "timestamp"])
        .groupby("user_id")["item_id"]
        .apply(list)
    )

    popularity = compute_item_popularity(grouped.tolist())
    popularity_group = build_popularity_groups(
        popularity,
        head_ratio=cfg["popularity"]["head_ratio"],
        mid_ratio=cfg["popularity"]["mid_ratio"],
    )

    popularity_df = pd.DataFrame(
        {
            "item_id": list(popularity.keys()),
            "interaction_count": list(popularity.values()),
            "popularity_group": [popularity_group[item_id] for item_id in popularity.keys()],
        }
    )

    items = items.copy()
    items["item_id"] = items["item_id"].astype(str)
    items = items.merge(
        popularity_df[["item_id", "popularity_group"]],
        on="item_id",
        how="left",
    )

    interactions.to_csv(processed_dir / "interactions.csv", index=False)
    items.to_csv(processed_dir / "items.csv", index=False)
    users.to_csv(processed_dir / "users.csv", index=False)
    popularity_df.to_csv(processed_dir / "popularity_stats.csv", index=False)

    print("[Done] preprocess finished")
    print(f"[Saved] {processed_dir / 'interactions.csv'}")
    print(f"[Saved] {processed_dir / 'items.csv'}")
    print(f"[Saved] {processed_dir / 'users.csv'}")
    print(f"[Saved] {processed_dir / 'popularity_stats.csv'}")
    print("[Stats]")
    print(stats)


if __name__ == "__main__":
    main()
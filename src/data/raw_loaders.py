from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _read_jsonl_dataframe(
    path: str | Path,
    columns: list[str] | None = None,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    path = Path(path)
    frames: list[pd.DataFrame] = []

    reader = pd.read_json(
        path,
        lines=True,
        compression="infer",
        chunksize=chunksize,
    )

    for chunk in reader:
        if columns is not None:
            existing = [col for col in columns if col in chunk.columns]
            chunk = chunk[existing].copy()
            for col in columns:
                if col not in chunk.columns:
                    chunk[col] = None
            chunk = chunk[columns]
        frames.append(chunk)

    if not frames:
        if columns is None:
            return pd.DataFrame()
        return pd.DataFrame(columns=columns)

    return pd.concat(frames, ignore_index=True)


def read_reviews_jsonl_or_gz(path: str | Path) -> pd.DataFrame:
    return _read_jsonl_dataframe(
        path,
        columns=["user_id", "reviewerID", "parent_asin", "asin", "rating", "overall", "timestamp", "unixReviewTime"],
    )


def read_meta_jsonl_or_gz(path: str | Path) -> pd.DataFrame:
    return _read_jsonl_dataframe(
        path,
        columns=["parent_asin", "asin", "title", "categories", "description"],
    )


def _pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"None of the candidate columns exist: {candidates}")


def normalize_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 Amazon review 原始字段统一成:
    user_id, item_id, rating, timestamp

    关键修复点：
    不再把 asin / parent_asin 同时 rename 成 item_id，
    而是明确只选一个主键列，避免产生重复列名。
    """
    if df.empty:
        return pd.DataFrame(columns=["user_id", "item_id", "rating", "timestamp"])

    user_col = _pick_first_existing_column(df, ["user_id", "reviewerID"])
    item_col = _pick_first_existing_column(df, ["parent_asin", "asin"])
    rating_col = _pick_first_existing_column(df, ["rating", "overall"])
    time_col = _pick_first_existing_column(df, ["timestamp", "unixReviewTime"])

    out = df[[user_col, item_col, rating_col, time_col]].copy()
    out.columns = ["user_id", "item_id", "rating", "timestamp"]

    out["user_id"] = out["user_id"].astype(str)
    out["item_id"] = out["item_id"].astype(str)
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")

    out = out.dropna(subset=["user_id", "item_id", "rating", "timestamp"])
    out["timestamp"] = out["timestamp"].astype("int64")

    # 去重，避免重复交互影响后续统计
    out = out.drop_duplicates(subset=["user_id", "item_id", "timestamp"]).reset_index(drop=True)

    return out


def normalize_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 Amazon meta 原始字段统一成:
    item_id, title, categories, description
    """
    if df.empty:
        return pd.DataFrame(columns=["item_id", "title", "categories", "description"])

    item_col = _pick_first_existing_column(df, ["parent_asin", "asin"])

    out = pd.DataFrame()
    out["item_id"] = df[item_col].astype(str)

    if "title" in df.columns:
        out["title"] = df["title"]
    else:
        out["title"] = None

    if "categories" in df.columns:
        out["categories"] = df["categories"]
    else:
        out["categories"] = None

    if "description" in df.columns:
        out["description"] = df["description"]
    else:
        out["description"] = None

    out = out.dropna(subset=["item_id"])
    out = out.drop_duplicates(subset=["item_id"]).reset_index(drop=True)

    return out


def filter_positive_interactions(df: pd.DataFrame, rating_threshold: float) -> pd.DataFrame:
    return df[df["rating"] >= rating_threshold].copy()


def iterative_k_core_filter(
    interactions: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> pd.DataFrame:
    """
    迭代执行 user/item 双侧 k-core 过滤，直到稳定。
    """
    out = interactions.copy()

    while True:
        prev_len = len(out)

        user_counts = out["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index

        out = out[out["user_id"].isin(valid_users)].copy()

        item_counts = out["item_id"].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index

        out = out[out["item_id"].isin(valid_items)].copy()

        if len(out) == prev_len:
            break

    out = out.reset_index(drop=True)
    return out


def load_amazon_domain(
    domain_name: str,
    review_path: str | Path,
    meta_path: str | Path,
    rating_threshold: float,
    min_user_interactions: int,
    min_item_interactions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    review_df = read_reviews_jsonl_or_gz(review_path)
    meta_df = read_meta_jsonl_or_gz(meta_path)

    interactions = normalize_review_columns(review_df)
    items = normalize_meta_columns(meta_df)

    interactions = filter_positive_interactions(interactions, rating_threshold)
    interactions = iterative_k_core_filter(
        interactions=interactions,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )

    valid_item_ids = set(interactions["item_id"].unique())
    items = items[items["item_id"].isin(valid_item_ids)].copy().reset_index(drop=True)

    users = pd.DataFrame({"user_id": sorted(interactions["user_id"].unique())})

    stats = {
        "domain_name": domain_name,
        "num_interactions": int(len(interactions)),
        "num_users": int(interactions["user_id"].nunique()),
        "num_items": int(interactions["item_id"].nunique()),
    }

    return interactions, items, users, stats

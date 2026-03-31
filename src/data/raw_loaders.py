from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def _check_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def load_movielens_1m(raw_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    读取 MovieLens 1M 原始数据。

    参数
    ----------
    raw_dir : str | Path
        MovieLens 1M 解压后的目录，例如：
        data/raw/movielens_1m/ml-1m

    返回
    ----------
    ratings_df : pd.DataFrame
        列: user_id, item_id, rating, timestamp
    movies_df : pd.DataFrame
        列: item_id, title, genres
    users_df : pd.DataFrame
        列: user_id, gender, age, occupation, zip_code
    """
    raw_dir = Path(raw_dir)

    ratings_path = raw_dir / "ratings.dat"
    movies_path = raw_dir / "movies.dat"
    users_path = raw_dir / "users.dat"

    _check_exists(ratings_path)
    _check_exists(movies_path)
    _check_exists(users_path)

    ratings_df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )

    movies_df = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        header=None,
        names=["item_id", "title", "genres"],
        encoding="latin-1",
    )

    users_df = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )

    ratings_df["user_id"] = ratings_df["user_id"].astype(int)
    ratings_df["item_id"] = ratings_df["item_id"].astype(int)
    ratings_df["rating"] = ratings_df["rating"].astype(float)
    ratings_df["timestamp"] = ratings_df["timestamp"].astype(int)

    movies_df["item_id"] = movies_df["item_id"].astype(int)
    movies_df["title"] = movies_df["title"].fillna("").astype(str)
    movies_df["genres"] = movies_df["genres"].fillna("").astype(str)

    users_df["user_id"] = users_df["user_id"].astype(int)

    return ratings_df, movies_df, users_df
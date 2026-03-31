from __future__ import annotations

from typing import Dict

import pandas as pd


def _normalize_genres(genres: str, genre_sep: str = ", ") -> str:
    if not genres or genres == "(no genres listed)":
        return "unknown genre"
    return genres.replace("|", genre_sep)


def build_movielens_item_text(
    movies_df: pd.DataFrame,
    use_title: bool = True,
    use_genres: bool = True,
    genre_sep: str = ", ",
) -> pd.DataFrame:
    """
    为 MovieLens item 构造统一文本字段 item_text。

    输入列要求:
    - item_id
    - title
    - genres

    输出会在原表基础上新增:
    - clean_title
    - genre_text
    - item_text
    """
    required_cols = {"item_id", "title", "genres"}
    missing = required_cols - set(movies_df.columns)
    if missing:
        raise ValueError(f"movies_df is missing columns: {missing}")

    df = movies_df.copy()
    df["clean_title"] = df["title"].fillna("").astype(str).str.strip()
    df["genre_text"] = df["genres"].fillna("").astype(str).apply(
        lambda x: _normalize_genres(x, genre_sep=genre_sep)
    )

    def _compose_row(row: Dict) -> str:
        parts = []
        if use_title and row["clean_title"]:
            parts.append(f"Title: {row['clean_title']}")
        if use_genres and row["genre_text"]:
            parts.append(f"Genres: {row['genre_text']}")
        return " | ".join(parts).strip()

    df["item_text"] = df.apply(_compose_row, axis=1)
    return df
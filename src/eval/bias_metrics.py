# src/eval/bias_metrics.py

from __future__ import annotations

from typing import Dict

import pandas as pd


def compute_topk_exposure_distribution(
    ranked_df: pd.DataFrame,
    k: int = 10,
    user_col: str = "user_id",
    rank_col: str = "rank",
    popularity_col: str = "target_popularity_group"
) -> pd.DataFrame:
    if popularity_col not in ranked_df.columns:
        raise ValueError(f"Column `{popularity_col}` not found in dataframe.")

    topk = (
        ranked_df.sort_values([user_col, rank_col])
        .groupby(user_col, as_index=False, group_keys=False)
        .head(k)
        .copy()
    )

    dist = (
        topk[popularity_col]
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .value_counts(normalize=True)
        .rename("fraction")
        .reset_index()
        .rename(columns={"index": popularity_col})
    )
    return dist


def compute_bias_metrics(
    ranked_df: pd.DataFrame,
    k: int = 10,
    user_col: str = "user_id",
    rank_col: str = "rank",
    popularity_col: str = "target_popularity_group",
    item_col: str = "candidate_item_id"
) -> Dict[str, float]:
    if popularity_col not in ranked_df.columns:
        raise ValueError(f"Column `{popularity_col}` not found in dataframe.")
    if item_col not in ranked_df.columns:
        raise ValueError(f"Column `{item_col}` not found in dataframe.")

    topk = (
        ranked_df.sort_values([user_col, rank_col])
        .groupby(user_col, as_index=False, group_keys=False)
        .head(k)
        .copy()
    )

    pop = topk[popularity_col].fillna("unknown").astype(str).str.lower()

    head_ratio = float((pop == "head").mean()) if len(topk) else float("nan")
    tail_ratio = float((pop == "tail").mean()) if len(topk) else float("nan")

    tail_items = topk.loc[pop == "tail", item_col].nunique()
    all_tail_items = ranked_df.loc[
        ranked_df[popularity_col].fillna("unknown").astype(str).str.lower() == "tail",
        item_col
    ].nunique()

    if all_tail_items == 0:
        long_tail_coverage = 0.0
    else:
        long_tail_coverage = float(tail_items / all_tail_items)

    return {
        f"head_exposure_ratio@{k}": head_ratio,
        f"tail_exposure_ratio@{k}": tail_ratio,
        f"long_tail_coverage@{k}": long_tail_coverage,
        f"topk_total_items@{k}": int(len(topk)),
    }
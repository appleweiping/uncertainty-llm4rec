from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import pandas as pd


def compute_item_popularity(interactions: Iterable[Sequence[str]]) -> dict[str, int]:
    """
    从用户交互序列中统计每个 item 的出现次数。

    这个函数主要兼容你当前早期小样本 pipeline 的用法。
    输入示例:
        [
            ["i1", "i2", "i3"],
            ["i2", "i4"],
        ]

    返回示例:
        {
            "i1": 1,
            "i2": 2,
            "i3": 1,
            "i4": 1,
        }
    """
    counter: Counter[str] = Counter()
    for seq in interactions:
        counter.update(seq)
    return dict(counter)


def compute_item_popularity_from_df(
    interactions_df: pd.DataFrame,
    item_col: str = "item_id",
) -> pd.DataFrame:
    """
    从交互表中统计每个 item 的交互次数。

    参数
    ----------
    interactions_df : pd.DataFrame
        至少包含 item_col 列的交互表，例如:
        user_id, item_id, rating, timestamp
    item_col : str
        item id 所在列名，默认是 item_id

    返回
    ----------
    popularity_df : pd.DataFrame
        列:
        - item_id
        - interaction_count
    """
    if item_col not in interactions_df.columns:
        raise ValueError(f"interactions_df is missing required column: {item_col}")

    popularity_df = (
        interactions_df.groupby(item_col)
        .size()
        .reset_index(name="interaction_count")
        .rename(columns={item_col: "item_id"})
        .sort_values(["interaction_count", "item_id"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return popularity_df


def build_popularity_groups(
    popularity: dict[str, int],
    head_ratio: float = 0.2,
    mid_ratio: float = 0.6,
) -> dict[str, str]:
    """
    兼容旧接口：输入 dict[item_id, interaction_count]，
    按频次从高到低排序后切成 head / mid / tail。

    参数
    ----------
    popularity : dict[str, int]
        item -> interaction_count
    head_ratio : float
        head 区间占比
    mid_ratio : float
        head + mid 的累计占比。tail 为剩余部分。

    返回
    ----------
    groups : dict[str, str]
        item_id -> popularity_group
    """
    if not popularity:
        return {}

    if not (0 < head_ratio < mid_ratio < 1):
        raise ValueError("Require 0 < head_ratio < mid_ratio < 1")

    items = sorted(popularity.items(), key=lambda x: (-x[1], x[0]))
    n = len(items)

    head_end = max(1, int(n * head_ratio))
    mid_end = max(head_end + 1, int(n * mid_ratio))
    mid_end = min(mid_end, n)

    groups: dict[str, str] = {}
    for idx, (item_id, _) in enumerate(items):
        if idx < head_end:
            groups[item_id] = "head"
        elif idx < mid_end:
            groups[item_id] = "mid"
        else:
            groups[item_id] = "tail"
    return groups


def build_popularity_groups_df(
    popularity_df: pd.DataFrame,
    method: str = "quantile",
    head_ratio: float = 0.2,
    mid_ratio: float = 0.6,
    bucket_names: tuple[str, str, str] = ("tail", "mid", "head"),
) -> pd.DataFrame:
    """
    通用版 popularity 分桶函数，推荐后续在真实数据上统一使用。

    参数
    ----------
    popularity_df : pd.DataFrame
        至少包含:
        - item_id
        - interaction_count

    method : str
        分桶方式:
        - "ratio": 按排序后的固定比例切分
        - "quantile": 按排序后均分成三段，更适合真实数据统一处理

    head_ratio : float
        method="ratio" 时 head 的占比
    mid_ratio : float
        method="ratio" 时 head+mid 的累计占比

    bucket_names : tuple[str, str, str]
        默认输出为 ("tail", "mid", "head")
        这样按交互次数从低到高对应 tail -> mid -> head

    返回
    ----------
    result_df : pd.DataFrame
        列:
        - item_id
        - interaction_count
        - popularity_rank
        - popularity_group
    """
    required_cols = {"item_id", "interaction_count"}
    missing = required_cols - set(popularity_df.columns)
    if missing:
        raise ValueError(f"popularity_df is missing required columns: {missing}")

    if len(bucket_names) != 3:
        raise ValueError("bucket_names must contain exactly 3 names")

    df = popularity_df.copy()

    # 这里统一按交互次数从低到高排序，便于 tail -> mid -> head
    df = df.sort_values(["interaction_count", "item_id"], ascending=[True, True]).reset_index(drop=True)
    df["popularity_rank"] = range(1, len(df) + 1)

    n = len(df)
    if n == 0:
        df["popularity_group"] = []
        return df

    tail_name, mid_name, head_name = bucket_names

    if method == "ratio":
        if not (0 < head_ratio < mid_ratio < 1):
            raise ValueError("Require 0 < head_ratio < mid_ratio < 1 for ratio method")

        # 注意这里当前 df 是从低到高排序，所以要从后往前切 head
        head_size = max(1, int(n * head_ratio))
        mid_size = max(1, int(n * (mid_ratio - head_ratio)))
        tail_size = n - head_size - mid_size

        # 保证三段至少尽量合理
        if tail_size <= 0:
            tail_size = max(1, n - head_size - mid_size)
        if tail_size + mid_size + head_size != n:
            mid_size = max(1, n - head_size - tail_size)

        groups = []
        for idx in range(n):
            if idx < tail_size:
                groups.append(tail_name)
            elif idx < tail_size + mid_size:
                groups.append(mid_name)
            else:
                groups.append(head_name)

        df["popularity_group"] = groups

    elif method == "quantile":
        # 按排序位置均分成三段，简单稳定，适合现在先落地
        first_cut = n // 3
        second_cut = (2 * n) // 3

        groups = []
        for idx in range(n):
            if idx < first_cut:
                groups.append(tail_name)
            elif idx < second_cut:
                groups.append(mid_name)
            else:
                groups.append(head_name)

        df["popularity_group"] = groups

    else:
        raise ValueError(f"Unsupported method: {method}")

    return df


def build_popularity_stats_from_interactions(
    interactions_df: pd.DataFrame,
    item_col: str = "item_id",
    method: str = "quantile",
    head_ratio: float = 0.2,
    mid_ratio: float = 0.6,
    bucket_names: tuple[str, str, str] = ("tail", "mid", "head"),
) -> pd.DataFrame:
    """
    一步完成：
    interactions_df -> interaction_count -> popularity_rank -> popularity_group

    这是后面 main_preprocess.py 最推荐直接调用的接口。
    """
    popularity_df = compute_item_popularity_from_df(
        interactions_df=interactions_df,
        item_col=item_col,
    )
    popularity_stats_df = build_popularity_groups_df(
        popularity_df=popularity_df,
        method=method,
        head_ratio=head_ratio,
        mid_ratio=mid_ratio,
        bucket_names=bucket_names,
    )
    return popularity_stats_df
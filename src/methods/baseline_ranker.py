# src/methods/baseline_ranker.py

from __future__ import annotations

import pandas as pd


def add_baseline_score(
    df: pd.DataFrame,
    score_col: str = "calibrated_confidence",
    output_col: str = "baseline_score"
) -> pd.DataFrame:
    """
    Add baseline ranking score.
    Default baseline score = calibrated_confidence.
    """
    out = df.copy()
    if score_col not in out.columns:
        raise ValueError(f"Column `{score_col}` not found in dataframe.")

    out[output_col] = out[score_col].astype(float)
    return out


def rank_by_score(
    df: pd.DataFrame,
    user_col: str = "user_id",
    score_col: str = "baseline_score",
    rank_col: str = "rank"
) -> pd.DataFrame:
    """
    Rank candidates within each user group by descending score.
    """
    out = df.copy()
    if user_col not in out.columns:
        raise ValueError(f"Column `{user_col}` not found in dataframe.")
    if score_col not in out.columns:
        raise ValueError(f"Column `{score_col}` not found in dataframe.")

    out = out.sort_values(
        by=[user_col, score_col],
        ascending=[True, False]
    ).copy()

    out[rank_col] = out.groupby(user_col).cumcount() + 1
    return out
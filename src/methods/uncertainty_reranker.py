# src/methods/uncertainty_reranker.py

from __future__ import annotations

import pandas as pd


def add_uncertainty_aware_score(
    df: pd.DataFrame,
    confidence_col: str = "calibrated_confidence",
    uncertainty_col: str = "uncertainty",
    lambda_penalty: float = 0.5,
    output_col: str = "rerank_score"
) -> pd.DataFrame:
    """
    final_score = calibrated_confidence - lambda_penalty * uncertainty
    """
    out = df.copy()

    if confidence_col not in out.columns:
        raise ValueError(f"Column `{confidence_col}` not found in dataframe.")
    if uncertainty_col not in out.columns:
        raise ValueError(f"Column `{uncertainty_col}` not found in dataframe.")

    out[output_col] = (
        out[confidence_col].astype(float)
        - lambda_penalty * out[uncertainty_col].astype(float)
    )
    return out


def rank_by_rerank_score(
    df: pd.DataFrame,
    user_col: str = "user_id",
    score_col: str = "rerank_score",
    rank_col: str = "rank"
) -> pd.DataFrame:
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
# src/methods/uncertainty_reranker.py

from __future__ import annotations

import pandas as pd


def _sort_for_reranking(
    df: pd.DataFrame,
    user_col: str,
    score_col: str,
) -> pd.DataFrame:
    sort_cols = [user_col, score_col]
    ascending = [True, False]

    if "candidate_item_id" in df.columns:
        sort_cols.append("candidate_item_id")
        ascending.append(True)

    return df.sort_values(by=sort_cols, ascending=ascending, kind="mergesort").copy()


def add_uncertainty_aware_score(
    df: pd.DataFrame,
    confidence_col: str = "calibrated_confidence",
    uncertainty_col: str = "uncertainty",
    lambda_penalty: float = 0.5,
    output_col: str = "final_score"
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
    score_col: str = "final_score",
    rank_col: str = "rank"
) -> pd.DataFrame:
    out = df.copy()

    if user_col not in out.columns:
        raise ValueError(f"Column `{user_col}` not found in dataframe.")
    if score_col not in out.columns:
        raise ValueError(f"Column `{score_col}` not found in dataframe.")

    out = _sort_for_reranking(out, user_col=user_col, score_col=score_col)

    out[rank_col] = out.groupby(user_col).cumcount() + 1
    return out


def rerank_candidates(
    df: pd.DataFrame,
    user_col: str = "user_id",
    confidence_col: str = "calibrated_confidence",
    uncertainty_col: str = "uncertainty",
    lambda_penalty: float = 0.5,
    score_col: str = "final_score",
    rank_col: str = "rank",
) -> pd.DataFrame:
    scored = add_uncertainty_aware_score(
        df=df,
        confidence_col=confidence_col,
        uncertainty_col=uncertainty_col,
        lambda_penalty=lambda_penalty,
        output_col=score_col,
    )
    return rank_by_rerank_score(
        scored,
        user_col=user_col,
        score_col=score_col,
        rank_col=rank_col,
    )

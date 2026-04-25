from __future__ import annotations

import math

import pandas as pd


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _rank_signature(df: pd.DataFrame, rank_col: str = "rank") -> list[str]:
    return df.sort_values(rank_col)["candidate_item_id"].astype(str).tolist()


def _topk_signature(df: pd.DataFrame, k: int, rank_col: str = "rank") -> tuple[str, ...]:
    return tuple(df.sort_values(rank_col).head(k)["candidate_item_id"].astype(str).tolist())


def _kendall_tau_from_orders(base_order: list[str], rerank_order: list[str]) -> float:
    shared = [item for item in base_order if item in set(rerank_order)]
    n = len(shared)
    if n < 2:
        return 1.0

    rerank_pos = {item: idx for idx, item in enumerate(rerank_order)}
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            left = shared[i]
            right = shared[j]
            if rerank_pos[left] < rerank_pos[right]:
                concordant += 1
            else:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 1.0
    return float((concordant - discordant) / total)


def _spread(series: pd.Series) -> float:
    values = _safe_numeric(series).dropna()
    if values.empty:
        return float("nan")
    return float(values.max() - values.min())


def compute_rank_change_diagnostics(
    baseline_ranked: pd.DataFrame,
    rerank_ranked: pd.DataFrame,
    *,
    base_score_col: str,
    uncertainty_col: str,
    lambda_penalty: float,
    setting: str,
    normalization: str = "none",
    k: int = 10,
    user_col: str = "user_id",
) -> dict:
    key_cols = [user_col, "candidate_item_id"]
    merged = baseline_ranked[key_cols + ["rank"]].rename(columns={"rank": "baseline_rank"}).merge(
        rerank_ranked[key_cols + ["rank"]].rename(columns={"rank": "rerank_rank"}),
        on=key_cols,
        how="inner",
    )
    if merged.empty:
        raise ValueError("No shared rows between baseline and reranked data.")

    rank_change_rate = float((merged["baseline_rank"] != merged["rerank_rank"]).mean())

    base_groups = {user: group for user, group in baseline_ranked.groupby(user_col)}
    rerank_groups = {user: group for user, group in rerank_ranked.groupby(user_col)}
    topk_changed = []
    topk_order_changed = []
    kendalls = []
    for user, base_group in base_groups.items():
        rerank_group = rerank_groups.get(user)
        if rerank_group is None:
            continue
        local_k = min(k, len(base_group), len(rerank_group))
        topk_changed.append(
            set(_topk_signature(base_group, local_k)) != set(_topk_signature(rerank_group, local_k))
        )
        topk_order_changed.append(
            _topk_signature(base_group, local_k) != _topk_signature(rerank_group, local_k)
        )
        kendalls.append(
            _kendall_tau_from_orders(_rank_signature(base_group), _rank_signature(rerank_group))
        )

    base_values = _safe_numeric(baseline_ranked[base_score_col])
    uncertainty_values = _safe_numeric(baseline_ranked[uncertainty_col])
    spearman = base_values.corr(uncertainty_values, method="spearman")

    base_spreads = baseline_ranked.groupby(user_col)[base_score_col].apply(_spread)
    uncertainty_spreads = baseline_ranked.groupby(user_col)[uncertainty_col].apply(_spread)

    row = {
        "setting": setting,
        "lambda": float(lambda_penalty),
        "base_score_col": base_score_col,
        "uncertainty_col": uncertainty_col,
        "normalization": normalization,
        "rank_change_rate": rank_change_rate,
        "top10_change_rate": float(pd.Series(topk_changed).mean()) if topk_changed else float("nan"),
        "top10_order_change_rate": float(pd.Series(topk_order_changed).mean()) if topk_order_changed else float("nan"),
        "mean_kendall_tau": float(pd.Series(kendalls).mean()) if kendalls else float("nan"),
        "base_uncertainty_spearman": float(spearman) if not pd.isna(spearman) else float("nan"),
        "mean_base_score_spread": float(base_spreads.mean()),
        "mean_uncertainty_spread": float(uncertainty_spreads.mean()),
        "num_users": int(baseline_ranked[user_col].nunique()),
        "num_samples": int(len(baseline_ranked)),
    }
    row["rerank_is_noop"] = bool(
        math.isclose(row["rank_change_rate"], 0.0, abs_tol=1e-12)
        and math.isclose(row["top10_order_change_rate"], 0.0, abs_tol=1e-12)
    )
    return row


def add_evidence_risk(
    df: pd.DataFrame,
    *,
    output_col: str = "evidence_risk",
    alpha: float = 1.0 / 3.0,
    beta: float = 1.0 / 3.0,
    gamma: float = 1.0 / 3.0,
) -> pd.DataFrame:
    required = ["abs_evidence_margin", "ambiguity", "missing_information"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Cannot compute evidence_risk; missing columns: {missing}")

    out = df.copy()
    out[output_col] = (
        alpha * (1.0 - _safe_numeric(out["abs_evidence_margin"]).clip(0.0, 1.0))
        + beta * _safe_numeric(out["ambiguity"]).clip(0.0, 1.0)
        + gamma * _safe_numeric(out["missing_information"]).clip(0.0, 1.0)
    )
    return out


def add_user_normalized_column(
    df: pd.DataFrame,
    source_col: str,
    output_col: str,
    *,
    user_col: str = "user_id",
    method: str = "minmax",
    epsilon: float = 1e-12,
) -> pd.DataFrame:
    out = df.copy()
    values = _safe_numeric(out[source_col])

    if method == "none":
        out[output_col] = values
        return out

    if method == "minmax":
        grouped = values.groupby(out[user_col])
        mins = grouped.transform("min")
        maxs = grouped.transform("max")
        spread = maxs - mins
        out[output_col] = (values - mins) / spread.where(spread.abs() > epsilon, 1.0)
        out.loc[spread.abs() <= epsilon, output_col] = values.loc[spread.abs() <= epsilon]
        return out

    if method == "zscore":
        grouped = values.groupby(out[user_col])
        means = grouped.transform("mean")
        stds = grouped.transform("std").fillna(0.0)
        out[output_col] = (values - means) / stds.where(stds.abs() > epsilon, 1.0)
        out.loc[stds.abs() <= epsilon, output_col] = values.loc[stds.abs() <= epsilon]
        return out

    raise ValueError(f"Unsupported normalization method: {method}")

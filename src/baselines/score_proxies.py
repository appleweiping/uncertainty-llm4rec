from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def load_baseline_input(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline input file not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)

    raise ValueError(f"Unsupported baseline input format: {path.suffix}")


def infer_input_format(df: pd.DataFrame, explicit_input_format: str = "auto") -> str:
    input_format = str(explicit_input_format).strip().lower()
    if input_format in {"score_rows", "rank_rows", "grouped_scores"}:
        return input_format

    columns = set(df.columns)
    if {"user_id", "candidate_item_id", "score", "label"}.issubset(columns):
        return "score_rows"
    if {"user_id", "candidate_item_id", "rank", "label"}.issubset(columns):
        return "rank_rows"
    if {"user_id", "candidate_scores"}.issubset(columns):
        return "grouped_scores"

    raise ValueError(
        "Cannot infer baseline input format. Expected one of: "
        "`score_rows`, `rank_rows`, or `grouped_scores`."
    )


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_candidate_scores(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        out: list[dict[str, Any]] = []
        for row in value:
            if not isinstance(row, dict):
                continue
            item_id = str(
                row.get("item_id")
                or row.get("candidate_item_id")
                or row.get("id")
                or ""
            ).strip()
            if not item_id:
                continue
            out.append(
                {
                    "item_id": item_id,
                    "score": _safe_float(row.get("score"), default=float("nan")),
                    "label": int(row.get("label", 0)),
                    "reason": str(row.get("reason", "")).strip(),
                }
            )
        return out
    return []


def build_ranked_dataframe(
    raw_df: pd.DataFrame,
    input_format: str = "auto",
    user_col: str = "user_id",
    item_col: str = "candidate_item_id",
    label_col: str = "label",
    score_col: str = "score",
    rank_col: str = "rank",
) -> pd.DataFrame:
    resolved_format = infer_input_format(raw_df, explicit_input_format=input_format)

    if resolved_format == "score_rows":
        df = raw_df.copy()
        for required_col in [user_col, item_col, label_col, score_col]:
            if required_col not in df.columns:
                raise ValueError(f"Column `{required_col}` not found in score_rows input.")

        df[user_col] = df[user_col].astype(str).str.strip()
        df[item_col] = df[item_col].astype(str).str.strip()
        df[label_col] = df[label_col].astype(int)
        df[score_col] = df[score_col].astype(float)

        if "target_popularity_group" not in df.columns:
            df["target_popularity_group"] = "unknown"
        df["target_popularity_group"] = (
            df["target_popularity_group"].fillna("unknown").astype(str).str.lower()
        )

        df = df.sort_values(
            by=[user_col, score_col, item_col],
            ascending=[True, False, True],
            kind="mergesort",
        ).copy()
        df[rank_col] = df.groupby(user_col).cumcount() + 1
        return df.rename(
            columns={
                user_col: "user_id",
                item_col: "candidate_item_id",
                label_col: "label",
                score_col: "score",
                rank_col: "rank",
            }
        )

    if resolved_format == "rank_rows":
        df = raw_df.copy()
        for required_col in [user_col, item_col, label_col, rank_col]:
            if required_col not in df.columns:
                raise ValueError(f"Column `{required_col}` not found in rank_rows input.")

        df[user_col] = df[user_col].astype(str).str.strip()
        df[item_col] = df[item_col].astype(str).str.strip()
        df[label_col] = df[label_col].astype(int)
        df[rank_col] = df[rank_col].astype(int)
        if score_col not in df.columns:
            df[score_col] = float("nan")
        else:
            df[score_col] = df[score_col].astype(float)

        if "target_popularity_group" not in df.columns:
            df["target_popularity_group"] = "unknown"
        df["target_popularity_group"] = (
            df["target_popularity_group"].fillna("unknown").astype(str).str.lower()
        )

        return df.rename(
            columns={
                user_col: "user_id",
                item_col: "candidate_item_id",
                label_col: "label",
                score_col: "score",
                rank_col: "rank",
            }
        )

    rows: list[dict[str, Any]] = []
    for record in raw_df.to_dict(orient="records"):
        user_id = str(record.get("user_id", "")).strip()
        popularity_group = str(record.get("target_popularity_group", "unknown")).strip().lower() or "unknown"
        candidate_scores = _normalize_candidate_scores(record.get("candidate_scores"))
        sorted_scores = sorted(
            candidate_scores,
            key=lambda row: (float("-inf") if pd.isna(row["score"]) else row["score"]),
            reverse=True,
        )
        for rank, row in enumerate(sorted_scores, start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "candidate_item_id": row["item_id"],
                    "label": int(row.get("label", 0)),
                    "score": _safe_float(row.get("score")),
                    "rank": rank,
                    "target_popularity_group": popularity_group,
                }
            )

    return pd.DataFrame(rows)


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    if total <= 0:
        return [float("nan")] * len(values)
    return [value / total for value in exps]


def _normalized_entropy(probabilities: list[float]) -> float:
    if not probabilities:
        return float("nan")
    valid_probs = [prob for prob in probabilities if prob > 0]
    if not valid_probs:
        return 0.0
    entropy = -sum(prob * math.log(prob + 1e-12) for prob in valid_probs)
    normalizer = math.log(len(probabilities) + 1e-12)
    if normalizer <= 0:
        return 0.0
    return float(entropy / normalizer)


def build_proxy_rows(ranked_df: pd.DataFrame) -> pd.DataFrame:
    if ranked_df.empty:
        return pd.DataFrame(
            columns=[
                "user_id",
                "candidate_count",
                "top1_item_id",
                "top1_label",
                "top1_score",
                "top2_score",
                "score_margin",
                "proxy_confidence",
                "score_entropy",
                "score_sharpness",
                "target_popularity_group",
            ]
        )

    rows: list[dict[str, Any]] = []
    for user_id, user_df in ranked_df.groupby("user_id"):
        user_df = user_df.sort_values("rank").copy()
        scores = user_df["score"].astype(float).tolist()
        probs = _softmax(scores)
        top1 = user_df.iloc[0]
        top2_score = float(user_df.iloc[1]["score"]) if len(user_df) >= 2 else float("nan")
        top1_prob = probs[0] if probs else float("nan")
        entropy = _normalized_entropy(probs)

        rows.append(
            {
                "user_id": str(user_id),
                "candidate_count": int(len(user_df)),
                "top1_item_id": str(top1["candidate_item_id"]),
                "top1_label": int(top1["label"]),
                "top1_score": _safe_float(top1["score"]),
                "top2_score": _safe_float(top2_score),
                "score_margin": _safe_float(float(top1["score"]) - top2_score) if not pd.isna(top2_score) else float("nan"),
                "proxy_confidence": _safe_float(top1_prob),
                "score_entropy": _safe_float(entropy),
                "score_sharpness": _safe_float(1.0 - entropy) if not pd.isna(entropy) else float("nan"),
                "target_popularity_group": str(top1.get("target_popularity_group", "unknown")).strip().lower() or "unknown",
            }
        )

    return pd.DataFrame(rows)


def build_proxy_pointwise_df(proxy_df: pd.DataFrame) -> pd.DataFrame:
    if proxy_df.empty:
        return pd.DataFrame(columns=["recommend", "pred_label", "label", "confidence", "target_popularity_group"])

    df = proxy_df.copy()
    return pd.DataFrame(
        {
            "recommend": ["yes"] * len(df),
            "pred_label": [1] * len(df),
            "label": df["top1_label"].astype(int),
            "confidence": df["proxy_confidence"].astype(float).clip(0.0, 1.0),
            "target_popularity_group": df["target_popularity_group"].astype(str),
        }
    )


def compute_proxy_bin_rows(proxy_df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    if proxy_df.empty:
        return pd.DataFrame(
            columns=["bin_lower", "bin_upper", "bin_center", "count", "avg_confidence", "accuracy", "avg_margin"]
        )

    rows: list[dict[str, Any]] = []
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    for idx in range(n_bins):
        lower = bin_edges[idx]
        upper = bin_edges[idx + 1]
        if idx == n_bins - 1:
            subset = proxy_df[(proxy_df["proxy_confidence"] >= lower) & (proxy_df["proxy_confidence"] <= upper)].copy()
        else:
            subset = proxy_df[(proxy_df["proxy_confidence"] >= lower) & (proxy_df["proxy_confidence"] < upper)].copy()

        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "bin_center": float((lower + upper) / 2.0),
                "count": int(len(subset)),
                "avg_confidence": float(subset["proxy_confidence"].mean()) if len(subset) else float("nan"),
                "accuracy": float(subset["top1_label"].mean()) if len(subset) else float("nan"),
                "avg_margin": float(subset["score_margin"].mean()) if len(subset) else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def compute_proxy_popularity_rows(
    proxy_df: pd.DataFrame,
    high_conf_threshold: float = 0.8,
) -> pd.DataFrame:
    if proxy_df.empty:
        return pd.DataFrame(
            columns=[
                "target_popularity_group",
                "num_users",
                "avg_proxy_confidence",
                "avg_accuracy",
                "avg_margin",
                "avg_entropy",
                "high_conf_fraction",
                "wrong_high_conf_fraction",
            ]
        )

    rows: list[dict[str, Any]] = []
    for group_name, group_df in proxy_df.groupby("target_popularity_group"):
        rows.append(
            {
                "target_popularity_group": str(group_name),
                "num_users": int(len(group_df)),
                "avg_proxy_confidence": float(group_df["proxy_confidence"].mean()),
                "avg_accuracy": float(group_df["top1_label"].mean()),
                "avg_margin": float(group_df["score_margin"].mean()) if group_df["score_margin"].notna().any() else float("nan"),
                "avg_entropy": float(group_df["score_entropy"].mean()) if group_df["score_entropy"].notna().any() else float("nan"),
                "high_conf_fraction": float((group_df["proxy_confidence"] >= high_conf_threshold).mean()),
                "wrong_high_conf_fraction": float(
                    ((group_df["top1_label"] == 0) & (group_df["proxy_confidence"] >= high_conf_threshold)).mean()
                ),
            }
        )

    result = pd.DataFrame(rows)
    order = {"head": 0, "mid": 1, "tail": 2, "unknown": 3}
    if not result.empty:
        result["sort_key"] = result["target_popularity_group"].map(order).fillna(999)
        result = result.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)
    return result


def dump_minimal_example_json(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "user_id": "u1",
            "candidate_item_id": "i_pos_u1",
            "score": 3.2,
            "label": 1,
            "target_popularity_group": "mid",
        },
        {
            "user_id": "u1",
            "candidate_item_id": "i_neg1_u1",
            "score": 1.1,
            "label": 0,
            "target_popularity_group": "mid",
        },
        {
            "user_id": "u1",
            "candidate_item_id": "i_neg2_u1",
            "score": 0.4,
            "label": 0,
            "target_popularity_group": "mid",
        },
        {
            "user_id": "u2",
            "candidate_item_id": "i_neg1_u2",
            "score": 2.7,
            "label": 0,
            "target_popularity_group": "tail",
        },
        {
            "user_id": "u2",
            "candidate_item_id": "i_pos_u2",
            "score": 2.2,
            "label": 1,
            "target_popularity_group": "tail",
        },
        {
            "user_id": "u2",
            "candidate_item_id": "i_neg2_u2",
            "score": 0.5,
            "label": 0,
            "target_popularity_group": "tail",
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

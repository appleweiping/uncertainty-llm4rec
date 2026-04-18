from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _normalize_item_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [text]


def _build_lookup(
    uncertainty_df: pd.DataFrame,
    *,
    user_col: str,
    item_col: str,
    uncertainty_col: str,
    uncertainty_confidence_col: str | None = None,
) -> tuple[dict[tuple[str, str], dict[str, float]], float]:
    required_cols = [user_col, item_col, uncertainty_col]
    for col in required_cols:
        if col not in uncertainty_df.columns:
            raise ValueError(f"Column `{col}` not found in uncertainty dataframe.")

    lookup: dict[tuple[str, str], dict[str, float]] = {}
    uncertainties: list[float] = []

    for record in uncertainty_df.to_dict(orient="records"):
        key = (str(record[user_col]), str(record[item_col]))
        uncertainty_value = float(record[uncertainty_col])
        entry = {"uncertainty": uncertainty_value}
        if uncertainty_confidence_col and uncertainty_confidence_col in record and pd.notna(record[uncertainty_confidence_col]):
            entry["uncertainty_confidence"] = float(record[uncertainty_confidence_col])
        lookup[key] = entry
        uncertainties.append(uncertainty_value)

    fallback_uncertainty = float(np.mean(uncertainties)) if uncertainties else 0.5
    return lookup, fallback_uncertainty


def _position_to_relevance(rank_position: int, num_candidates: int) -> float:
    if num_candidates <= 0:
        return 0.0
    return float((num_candidates - rank_position + 1) / num_candidates)


def build_ranker_rows(
    ranking_predictions_df: pd.DataFrame,
    uncertainty_df: pd.DataFrame,
    *,
    lambda_penalty: float,
    topk: int = 10,
    user_col: str = "user_id",
    item_col: str = "candidate_item_id",
    uncertainty_col: str = "uncertainty",
    uncertainty_confidence_col: str | None = "calibrated_confidence",
    uncertainty_source: str = "pointwise_calibrated",
) -> pd.DataFrame:
    lookup, fallback_uncertainty = _build_lookup(
        uncertainty_df,
        user_col=user_col,
        item_col=item_col,
        uncertainty_col=uncertainty_col,
        uncertainty_confidence_col=uncertainty_confidence_col,
    )

    rows: list[dict[str, Any]] = []
    for record in ranking_predictions_df.to_dict(orient="records"):
        candidate_ids = _normalize_item_list(record.get("candidate_item_ids"))
        ranked_ids = _normalize_item_list(record.get("pred_ranked_item_ids"))
        popularity_groups = _normalize_item_list(record.get("candidate_popularity_groups"))
        candidate_popularity = {
            item_id: (
                str(popularity_groups[idx]).strip().lower()
                if idx < len(popularity_groups) and str(popularity_groups[idx]).strip()
                else "unknown"
            )
            for idx, item_id in enumerate(candidate_ids)
        }

        rank_lookup = {item_id: idx + 1 for idx, item_id in enumerate(ranked_ids)}
        num_candidates = len(candidate_ids)
        positive_item_id = str(record.get("positive_item_id", "")).strip()

        for item_id in candidate_ids:
            original_rank = int(rank_lookup.get(item_id, num_candidates + 1))
            relevance_score = _position_to_relevance(original_rank, num_candidates)
            uncertainty_key = (str(record.get("user_id")), str(item_id))
            uncertainty_entry = lookup.get(uncertainty_key)
            if uncertainty_entry is None:
                candidate_uncertainty = fallback_uncertainty
                uncertainty_confidence = float("nan")
                matched_uncertainty = False
            else:
                candidate_uncertainty = float(uncertainty_entry["uncertainty"])
                uncertainty_confidence = float(uncertainty_entry.get("uncertainty_confidence", np.nan))
                matched_uncertainty = True

            final_score = relevance_score - float(lambda_penalty) * candidate_uncertainty

            rows.append(
                {
                    "user_id": record.get("user_id"),
                    "source_event_id": record.get("source_event_id"),
                    "split_name": record.get("split_name"),
                    "timestamp": record.get("timestamp"),
                    "positive_item_id": positive_item_id,
                    "candidate_item_id": str(item_id),
                    "candidate_popularity_group": candidate_popularity.get(item_id, "unknown"),
                    "label": int(str(item_id) == positive_item_id) if positive_item_id else 0,
                    "num_candidates": int(num_candidates),
                    "original_rank": original_rank,
                    "relevance_score": relevance_score,
                    "uncertainty": candidate_uncertainty,
                    "uncertainty_confidence": uncertainty_confidence,
                    "matched_uncertainty": matched_uncertainty,
                    "final_score": final_score,
                    "uncertainty_source": uncertainty_source,
                    "lambda_penalty": float(lambda_penalty),
                    "topk": int(topk),
                    "raw_rank_confidence": float(record.get("confidence", np.nan)),
                    "rank_parse_success": bool(record.get("parse_success", False)),
                    "rank_latency": float(record.get("latency", np.nan)),
                    "raw_response": record.get("raw_response"),
                }
            )

    return pd.DataFrame(rows)


def rank_candidates_by_score(
    scored_df: pd.DataFrame,
    *,
    group_col: str = "source_event_id",
    score_col: str = "final_score",
    rank_col: str = "rerank_rank",
) -> pd.DataFrame:
    if group_col not in scored_df.columns:
        raise ValueError(f"Column `{group_col}` not found in scored dataframe.")
    if score_col not in scored_df.columns:
        raise ValueError(f"Column `{score_col}` not found in scored dataframe.")

    out = scored_df.sort_values(
        by=[group_col, score_col, "candidate_item_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    out[rank_col] = out.groupby(group_col).cumcount() + 1
    return out


def build_reranked_predictions(
    scored_ranked_df: pd.DataFrame,
    original_predictions_df: pd.DataFrame,
    *,
    topk: int = 10,
) -> pd.DataFrame:
    original_lookup = {
        str(record.get("source_event_id")): record
        for record in original_predictions_df.to_dict(orient="records")
    }

    rows: list[dict[str, Any]] = []
    for source_event_id, event_df in scored_ranked_df.groupby("source_event_id", dropna=False):
        event_df = event_df.sort_values("rerank_rank").copy()
        original_record = original_lookup.get(str(source_event_id), {})

        reranked_item_ids = event_df["candidate_item_id"].astype(str).tolist()
        topk_item_ids = reranked_item_ids[:topk]
        candidate_scores = {
            str(row["candidate_item_id"]): {
                "original_rank": int(row["original_rank"]),
                "relevance_score": float(row["relevance_score"]),
                "uncertainty": float(row["uncertainty"]),
                "final_score": float(row["final_score"]),
                "matched_uncertainty": bool(row["matched_uncertainty"]),
            }
            for _, row in event_df.iterrows()
        }
        missing_uncertainty_item_ids = [
            str(row["candidate_item_id"])
            for _, row in event_df.iterrows()
            if not bool(row["matched_uncertainty"])
        ]

        rows.append(
            {
                "user_id": event_df.iloc[0]["user_id"],
                "source_event_id": source_event_id,
                "positive_item_id": event_df.iloc[0]["positive_item_id"],
                "split_name": event_df.iloc[0]["split_name"],
                "timestamp": event_df.iloc[0]["timestamp"],
                "candidate_item_ids": _normalize_item_list(original_record.get("candidate_item_ids")),
                "candidate_titles": _normalize_item_list(original_record.get("candidate_titles")),
                "candidate_popularity_groups": _normalize_item_list(original_record.get("candidate_popularity_groups")),
                "original_pred_ranked_item_ids": _normalize_item_list(original_record.get("pred_ranked_item_ids")),
                "pred_ranked_item_ids": reranked_item_ids,
                "topk_item_ids": topk_item_ids,
                "confidence": float(original_record.get("confidence", np.nan)),
                "parse_success": bool(original_record.get("parse_success", False)),
                "latency": float(original_record.get("latency", np.nan)),
                "contains_out_of_candidate_item": bool(original_record.get("contains_out_of_candidate_item", False)),
                "out_of_candidate_item_ids": _normalize_item_list(original_record.get("out_of_candidate_item_ids")),
                "candidate_scores": candidate_scores,
                "uncertainty_source": event_df.iloc[0]["uncertainty_source"],
                "lambda_penalty": float(event_df.iloc[0]["lambda_penalty"]),
                "uncertainty_coverage_rate": float(event_df["matched_uncertainty"].mean()),
                "missing_uncertainty_item_ids": missing_uncertainty_item_ids,
                "raw_response": original_record.get("raw_response"),
            }
        )

    return pd.DataFrame(rows)


def summarize_rerank_effect(
    original_predictions_df: pd.DataFrame,
    reranked_predictions_df: pd.DataFrame,
) -> dict[str, float]:
    original_lookup = {
        str(record.get("source_event_id")): _normalize_item_list(record.get("pred_ranked_item_ids"))
        for record in original_predictions_df.to_dict(orient="records")
    }

    changed = 0
    total = 0
    average_shift_accumulator = []
    coverage_values = []

    for record in reranked_predictions_df.to_dict(orient="records"):
        event_id = str(record.get("source_event_id"))
        old_rank = original_lookup.get(event_id, [])
        new_rank = _normalize_item_list(record.get("pred_ranked_item_ids"))
        if not new_rank:
            continue
        total += 1
        if new_rank != old_rank:
            changed += 1

        old_positions = {item_id: idx + 1 for idx, item_id in enumerate(old_rank)}
        shifts = [abs((idx + 1) - old_positions.get(item_id, idx + 1)) for idx, item_id in enumerate(new_rank)]
        average_shift_accumulator.append(float(np.mean(shifts)) if shifts else 0.0)
        coverage_values.append(float(record.get("uncertainty_coverage_rate", np.nan)))

    return {
        "changed_ranking_fraction": float(changed / total) if total else float("nan"),
        "avg_position_shift": float(np.mean(average_shift_accumulator)) if average_shift_accumulator else float("nan"),
        "avg_uncertainty_coverage_rate": float(np.nanmean(coverage_values)) if coverage_values else float("nan"),
    }

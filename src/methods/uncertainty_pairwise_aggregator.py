from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_PAIRWISE_AGG_VARIANTS = {
    "weighted_win_count",
    "weighted_borda",
}


def _normalize_item_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [text]


def _build_uncertainty_lookup(
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
        entry = {"uncertainty": float(record[uncertainty_col])}
        if (
            uncertainty_confidence_col
            and uncertainty_confidence_col in record
            and pd.notna(record[uncertainty_confidence_col])
        ):
            entry["uncertainty_confidence"] = float(record[uncertainty_confidence_col])
        lookup[key] = entry
        uncertainties.append(float(record[uncertainty_col]))

    fallback_uncertainty = float(np.mean(uncertainties)) if uncertainties else 0.5
    return lookup, fallback_uncertainty


def _position_to_relevance(rank_position: int, num_candidates: int) -> float:
    if num_candidates <= 0:
        return 0.0
    return float((num_candidates - rank_position + 1) / num_candidates)


def _build_event_contexts(
    ranking_predictions_df: pd.DataFrame,
    uncertainty_lookup: dict[tuple[str, str], dict[str, float]],
    *,
    fallback_uncertainty: float,
) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}

    for record in ranking_predictions_df.to_dict(orient="records"):
        source_event_id = str(record.get("source_event_id"))
        user_id = str(record.get("user_id"))
        candidate_ids = _normalize_item_list(record.get("candidate_item_ids"))
        ranked_ids = _normalize_item_list(record.get("pred_ranked_item_ids"))
        popularity_groups = _normalize_item_list(record.get("candidate_popularity_groups"))
        candidate_titles = _normalize_item_list(record.get("candidate_titles"))

        candidate_popularity = {
            item_id: (
                str(popularity_groups[idx]).strip().lower()
                if idx < len(popularity_groups) and str(popularity_groups[idx]).strip()
                else "unknown"
            )
            for idx, item_id in enumerate(candidate_ids)
        }
        candidate_title_lookup = {
            item_id: candidate_titles[idx] if idx < len(candidate_titles) else ""
            for idx, item_id in enumerate(candidate_ids)
        }

        rank_lookup = {item_id: idx + 1 for idx, item_id in enumerate(ranked_ids)}
        matched_uncertainties: list[float] = []
        item_uncertainties: dict[str, float] = {}
        item_uncertainty_confidences: dict[str, float] = {}
        matched_uncertainty_flags: dict[str, bool] = {}

        for item_id in candidate_ids:
            uncertainty_key = (user_id, str(item_id))
            uncertainty_entry = uncertainty_lookup.get(uncertainty_key)
            if uncertainty_entry is None:
                item_uncertainties[str(item_id)] = float("nan")
                item_uncertainty_confidences[str(item_id)] = float("nan")
                matched_uncertainty_flags[str(item_id)] = False
            else:
                uncertainty_value = float(uncertainty_entry["uncertainty"])
                item_uncertainties[str(item_id)] = uncertainty_value
                item_uncertainty_confidences[str(item_id)] = float(
                    uncertainty_entry.get("uncertainty_confidence", np.nan)
                )
                matched_uncertainty_flags[str(item_id)] = True
                matched_uncertainties.append(uncertainty_value)

        event_mean_uncertainty = (
            float(np.mean(matched_uncertainties))
            if matched_uncertainties
            else float(fallback_uncertainty)
        )
        event_uncertainty_coverage_rate = (
            float(len(matched_uncertainties) / len(candidate_ids)) if candidate_ids else 0.0
        )

        for item_id in candidate_ids:
            if not np.isnan(item_uncertainties[str(item_id)]):
                continue
            item_uncertainties[str(item_id)] = event_mean_uncertainty

        contexts[source_event_id] = {
            "record": record,
            "user_id": user_id,
            "candidate_ids": candidate_ids,
            "candidate_titles": candidate_titles,
            "candidate_title_lookup": candidate_title_lookup,
            "candidate_popularity": candidate_popularity,
            "rank_lookup": rank_lookup,
            "pred_ranked_item_ids": ranked_ids,
            "event_mean_uncertainty": event_mean_uncertainty,
            "event_uncertainty_coverage_rate": event_uncertainty_coverage_rate,
            "item_uncertainties": item_uncertainties,
            "item_uncertainty_confidences": item_uncertainty_confidences,
            "matched_uncertainty_flags": matched_uncertainty_flags,
            "positive_item_id": str(record.get("positive_item_id", "")).strip(),
        }

    return contexts


def build_pairwise_preference_rows(
    pairwise_predictions_df: pd.DataFrame,
    ranking_predictions_df: pd.DataFrame,
    uncertainty_df: pd.DataFrame,
    *,
    user_col: str = "user_id",
    item_col: str = "candidate_item_id",
    uncertainty_col: str = "uncertainty",
    uncertainty_confidence_col: str | None = "calibrated_confidence",
    uncertainty_source: str = "pointwise_calibrated",
) -> pd.DataFrame:
    uncertainty_lookup, fallback_uncertainty = _build_uncertainty_lookup(
        uncertainty_df,
        user_col=user_col,
        item_col=item_col,
        uncertainty_col=uncertainty_col,
        uncertainty_confidence_col=uncertainty_confidence_col,
    )
    event_contexts = _build_event_contexts(
        ranking_predictions_df,
        uncertainty_lookup,
        fallback_uncertainty=fallback_uncertainty,
    )

    rows: list[dict[str, Any]] = []
    for record in pairwise_predictions_df.to_dict(orient="records"):
        source_event_id = str(record.get("source_event_id"))
        context = event_contexts.get(source_event_id)
        if context is None:
            continue

        item_a_id = str(record.get("item_a_id", "")).strip()
        item_b_id = str(record.get("item_b_id", "")).strip()
        preferred_item_pred = str(record.get("preferred_item_pred", "")).strip()
        parse_success = bool(record.get("parse_success", False))
        valid_preference = preferred_item_pred in {item_a_id, item_b_id}
        if not valid_preference:
            continue

        non_preferred_item_pred = item_b_id if preferred_item_pred == item_a_id else item_a_id
        item_a_uncertainty = float(context["item_uncertainties"].get(item_a_id, context["event_mean_uncertainty"]))
        item_b_uncertainty = float(context["item_uncertainties"].get(item_b_id, context["event_mean_uncertainty"]))
        pair_uncertainty = float(np.mean([item_a_uncertainty, item_b_uncertainty]))
        pair_reliability_weight = float(np.clip(1.0 - pair_uncertainty, 0.0, 1.0))

        rank_lookup = context["rank_lookup"]
        candidate_ids = context["candidate_ids"]
        total_possible_pairs = int(len(candidate_ids) * max(len(candidate_ids) - 1, 0) / 2)

        rows.append(
            {
                "pair_id": record.get("pair_id"),
                "source_event_id": source_event_id,
                "user_id": context["user_id"],
                "split_name": record.get("split_name"),
                "timestamp": record.get("timestamp"),
                "positive_item_id": context["positive_item_id"],
                "num_candidates": int(len(candidate_ids)),
                "total_possible_pairs": total_possible_pairs,
                "item_a_id": item_a_id,
                "item_b_id": item_b_id,
                "item_a_title": context["candidate_title_lookup"].get(item_a_id, ""),
                "item_b_title": context["candidate_title_lookup"].get(item_b_id, ""),
                "item_a_popularity_group": context["candidate_popularity"].get(item_a_id, "unknown"),
                "item_b_popularity_group": context["candidate_popularity"].get(item_b_id, "unknown"),
                "preferred_item_true": str(record.get("preferred_item_true", "")).strip(),
                "preferred_item_pred": preferred_item_pred,
                "non_preferred_item_pred": non_preferred_item_pred,
                "pair_type": str(record.get("pair_type", "")).strip(),
                "pair_confidence": float(record.get("confidence", np.nan)),
                "pair_reason": record.get("reason"),
                "pair_parse_success": parse_success,
                "item_a_uncertainty": item_a_uncertainty,
                "item_b_uncertainty": item_b_uncertainty,
                "item_a_uncertainty_confidence": float(
                    context["item_uncertainty_confidences"].get(item_a_id, np.nan)
                ),
                "item_b_uncertainty_confidence": float(
                    context["item_uncertainty_confidences"].get(item_b_id, np.nan)
                ),
                "item_a_matched_uncertainty": bool(context["matched_uncertainty_flags"].get(item_a_id, False)),
                "item_b_matched_uncertainty": bool(context["matched_uncertainty_flags"].get(item_b_id, False)),
                "pair_uncertainty": pair_uncertainty,
                "pair_reliability_weight": pair_reliability_weight,
                "base_rank_item_a": int(rank_lookup.get(item_a_id, len(candidate_ids) + 1)),
                "base_rank_item_b": int(rank_lookup.get(item_b_id, len(candidate_ids) + 1)),
                "event_mean_uncertainty": float(context["event_mean_uncertainty"]),
                "event_uncertainty_coverage_rate": float(context["event_uncertainty_coverage_rate"]),
                "uncertainty_source": uncertainty_source,
            }
        )

    return pd.DataFrame(rows)


def aggregate_pairwise_preferences(
    pairwise_rows_df: pd.DataFrame,
    ranking_predictions_df: pd.DataFrame,
    *,
    aggregation_variant: str = "weighted_win_count",
    prior_weight: float = 0.2,
    loss_weight: float = 1.0,
    score_scale: float = 1.0,
) -> pd.DataFrame:
    if aggregation_variant not in SUPPORTED_PAIRWISE_AGG_VARIANTS:
        raise ValueError(f"Unsupported aggregation variant: {aggregation_variant}")

    if pairwise_rows_df.empty:
        return pd.DataFrame()

    ranking_lookup = {
        str(record.get("source_event_id")): record
        for record in ranking_predictions_df.to_dict(orient="records")
    }

    rows: list[dict[str, Any]] = []
    for source_event_id, event_pair_df in pairwise_rows_df.groupby("source_event_id", dropna=False):
        ranking_record = ranking_lookup.get(str(source_event_id))
        if ranking_record is None:
            continue

        candidate_ids = _normalize_item_list(ranking_record.get("candidate_item_ids"))
        candidate_titles = _normalize_item_list(ranking_record.get("candidate_titles"))
        popularity_groups = _normalize_item_list(ranking_record.get("candidate_popularity_groups"))
        base_ranked_ids = _normalize_item_list(ranking_record.get("pred_ranked_item_ids"))
        base_rank_lookup = {item_id: idx + 1 for idx, item_id in enumerate(base_ranked_ids)}
        num_candidates = len(candidate_ids)
        max_pairs = int(num_candidates * max(num_candidates - 1, 0) / 2)

        title_lookup = {
            item_id: candidate_titles[idx] if idx < len(candidate_titles) else ""
            for idx, item_id in enumerate(candidate_ids)
        }
        popularity_lookup = {
            item_id: (
                str(popularity_groups[idx]).strip().lower()
                if idx < len(popularity_groups) and str(popularity_groups[idx]).strip()
                else "unknown"
            )
            for idx, item_id in enumerate(candidate_ids)
        }

        item_stats = {
            item_id: {
                "wins": 0.0,
                "losses": 0.0,
                "comparisons": 0,
                "comparison_weight_sum": 0.0,
                "uncertainty": float("nan"),
                "matched_uncertainty": False,
                "uncertainty_confidence": float("nan"),
            }
            for item_id in candidate_ids
        }

        for pair_record in event_pair_df.to_dict(orient="records"):
            winner = str(pair_record["preferred_item_pred"])
            loser = str(pair_record["non_preferred_item_pred"])
            weight = float(pair_record["pair_reliability_weight"])

            if winner in item_stats:
                item_stats[winner]["wins"] += weight
                item_stats[winner]["comparisons"] += 1
                item_stats[winner]["comparison_weight_sum"] += weight
            if loser in item_stats:
                item_stats[loser]["losses"] += weight
                item_stats[loser]["comparisons"] += 1
                item_stats[loser]["comparison_weight_sum"] += weight

            for item_key, uncertainty_key, confidence_key, matched_key in [
                (str(pair_record["item_a_id"]), "item_a_uncertainty", "item_a_uncertainty_confidence", "item_a_matched_uncertainty"),
                (str(pair_record["item_b_id"]), "item_b_uncertainty", "item_b_uncertainty_confidence", "item_b_matched_uncertainty"),
            ]:
                if item_key not in item_stats:
                    continue
                item_stats[item_key]["uncertainty"] = float(pair_record[uncertainty_key])
                item_stats[item_key]["uncertainty_confidence"] = float(pair_record[confidence_key])
                item_stats[item_key]["matched_uncertainty"] = bool(pair_record[matched_key])

        supported_candidates = {
            str(item_id)
            for pair_record in event_pair_df.to_dict(orient="records")
            for item_id in [pair_record["item_a_id"], pair_record["item_b_id"]]
        }

        for item_id in candidate_ids:
            stats = item_stats[item_id]
            base_rank = int(base_rank_lookup.get(item_id, num_candidates + 1))
            base_relevance = _position_to_relevance(base_rank, num_candidates)
            comparisons = int(stats["comparisons"])
            weighted_margin = float(stats["wins"] - float(loss_weight) * stats["losses"])
            if aggregation_variant == "weighted_borda" and comparisons > 0:
                pairwise_score = float(weighted_margin / comparisons)
            else:
                pairwise_score = float(weighted_margin)

            prior_score = float(prior_weight) * base_relevance
            final_score = float(score_scale) * pairwise_score + prior_score

            rows.append(
                {
                    "user_id": ranking_record.get("user_id"),
                    "source_event_id": source_event_id,
                    "split_name": ranking_record.get("split_name"),
                    "timestamp": ranking_record.get("timestamp"),
                    "positive_item_id": str(ranking_record.get("positive_item_id", "")).strip(),
                    "candidate_item_id": str(item_id),
                    "candidate_title": title_lookup.get(item_id, ""),
                    "candidate_popularity_group": popularity_lookup.get(item_id, "unknown"),
                    "label": int(str(item_id) == str(ranking_record.get("positive_item_id", "")).strip()),
                    "num_candidates": int(num_candidates),
                    "base_rank": base_rank,
                    "base_relevance": base_relevance,
                    "wins": float(stats["wins"]),
                    "losses": float(stats["losses"]),
                    "comparisons": comparisons,
                    "comparison_weight_sum": float(stats["comparison_weight_sum"]),
                    "pairwise_score": pairwise_score,
                    "prior_score": prior_score,
                    "final_score": final_score,
                    "uncertainty": float(stats["uncertainty"]),
                    "uncertainty_confidence": float(stats["uncertainty_confidence"]),
                    "matched_uncertainty": bool(stats["matched_uncertainty"]),
                    "aggregation_variant": aggregation_variant,
                    "prior_weight": float(prior_weight),
                    "loss_weight": float(loss_weight),
                    "score_scale": float(score_scale),
                    "pairwise_pair_count": int(len(event_pair_df)),
                    "pairwise_pair_coverage_rate": float(len(event_pair_df) / max_pairs) if max_pairs else 0.0,
                    "pairwise_supported_candidate_fraction": (
                        float(len(supported_candidates) / num_candidates) if num_candidates else 0.0
                    ),
                    "pairwise_avg_reliability_weight": float(event_pair_df["pair_reliability_weight"].mean()),
                    "pairwise_parse_success_rate": float(event_pair_df["pair_parse_success"].mean()),
                    "pairwise_accuracy": float(
                        (
                            event_pair_df["preferred_item_true"].astype(str)
                            == event_pair_df["preferred_item_pred"].astype(str)
                        ).mean()
                    ),
                    "event_uncertainty_coverage_rate": float(event_pair_df["event_uncertainty_coverage_rate"].mean()),
                    "uncertainty_source": str(event_pair_df["uncertainty_source"].iloc[0]),
                }
            )

    return pd.DataFrame(rows)


def rank_pairwise_candidates(
    aggregated_item_df: pd.DataFrame,
    *,
    group_col: str = "source_event_id",
    score_col: str = "final_score",
    base_rank_col: str = "base_rank",
    rank_col: str = "pairwise_rank",
) -> pd.DataFrame:
    if aggregated_item_df.empty:
        return aggregated_item_df.copy()

    out = aggregated_item_df.sort_values(
        by=[group_col, score_col, base_rank_col, "candidate_item_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).copy()
    out[rank_col] = out.groupby(group_col).cumcount() + 1
    return out


def build_pairwise_ranked_predictions(
    ranked_item_df: pd.DataFrame,
    ranking_predictions_df: pd.DataFrame,
    *,
    topk: int = 10,
    total_ranking_events: int | None = None,
) -> pd.DataFrame:
    if ranked_item_df.empty:
        return pd.DataFrame()

    ranking_lookup = {
        str(record.get("source_event_id")): record
        for record in ranking_predictions_df.to_dict(orient="records")
    }

    rows: list[dict[str, Any]] = []
    for source_event_id, event_df in ranked_item_df.groupby("source_event_id", dropna=False):
        ranking_record = ranking_lookup.get(str(source_event_id), {})
        event_df = event_df.sort_values("pairwise_rank").copy()

        reranked_item_ids = event_df["candidate_item_id"].astype(str).tolist()
        topk_item_ids = reranked_item_ids[:topk]
        candidate_scores = {
            str(row["candidate_item_id"]): {
                "base_rank": int(row["base_rank"]),
                "base_relevance": float(row["base_relevance"]),
                "wins": float(row["wins"]),
                "losses": float(row["losses"]),
                "comparisons": int(row["comparisons"]),
                "pairwise_score": float(row["pairwise_score"]),
                "prior_score": float(row["prior_score"]),
                "final_score": float(row["final_score"]),
                "uncertainty": float(row["uncertainty"]),
                "matched_uncertainty": bool(row["matched_uncertainty"]),
            }
            for _, row in event_df.iterrows()
        }

        first_row = event_df.iloc[0]
        rows.append(
            {
                "user_id": first_row["user_id"],
                "source_event_id": source_event_id,
                "positive_item_id": first_row["positive_item_id"],
                "split_name": first_row["split_name"],
                "timestamp": first_row["timestamp"],
                "candidate_item_ids": _normalize_item_list(ranking_record.get("candidate_item_ids")),
                "candidate_titles": _normalize_item_list(ranking_record.get("candidate_titles")),
                "candidate_popularity_groups": _normalize_item_list(
                    ranking_record.get("candidate_popularity_groups")
                ),
                "original_pred_ranked_item_ids": _normalize_item_list(
                    ranking_record.get("pred_ranked_item_ids")
                ),
                "pred_ranked_item_ids": reranked_item_ids,
                "topk_item_ids": topk_item_ids,
                "confidence": float(first_row["pairwise_avg_reliability_weight"]),
                "parse_success": bool(first_row["pairwise_parse_success_rate"] == 1.0),
                "latency": float(ranking_record.get("latency", np.nan)),
                "contains_out_of_candidate_item": False,
                "out_of_candidate_item_ids": [],
                "candidate_scores": candidate_scores,
                "pairwise_exp_pair_count": int(first_row["pairwise_pair_count"]),
                "pairwise_pair_coverage_rate": float(first_row["pairwise_pair_coverage_rate"]),
                "pairwise_supported_candidate_fraction": float(
                    first_row["pairwise_supported_candidate_fraction"]
                ),
                "pairwise_avg_reliability_weight": float(first_row["pairwise_avg_reliability_weight"]),
                "pairwise_accuracy": float(first_row["pairwise_accuracy"]),
                "aggregation_variant": first_row["aggregation_variant"],
                "prior_weight": float(first_row["prior_weight"]),
                "loss_weight": float(first_row["loss_weight"]),
                "score_scale": float(first_row["score_scale"]),
                "uncertainty_source": first_row["uncertainty_source"],
                "uncertainty_coverage_rate": float(first_row["event_uncertainty_coverage_rate"]),
                "total_ranking_events": int(total_ranking_events) if total_ranking_events is not None else np.nan,
            }
        )

    return pd.DataFrame(rows)


def summarize_pairwise_rank_effect(
    original_predictions_df: pd.DataFrame,
    pairwise_ranked_predictions_df: pd.DataFrame,
    *,
    total_ranking_events: int | None = None,
) -> dict[str, float]:
    if pairwise_ranked_predictions_df.empty:
        return {
            "changed_ranking_fraction": float("nan"),
            "avg_position_shift": float("nan"),
            "avg_uncertainty_coverage_rate": float("nan"),
            "uncertainty_coverage": float("nan"),
            "pairwise_pair_coverage_rate": float("nan"),
            "pairwise_supported_candidate_fraction": float("nan"),
            "pairwise_avg_reliability_weight": float("nan"),
            "pairwise_supported_event_fraction": float("nan"),
        }

    original_lookup = {
        str(record.get("source_event_id")): _normalize_item_list(record.get("pred_ranked_item_ids"))
        for record in original_predictions_df.to_dict(orient="records")
    }

    changed = 0
    total = 0
    average_shift_accumulator: list[float] = []
    uncertainty_coverages: list[float] = []
    pairwise_coverages: list[float] = []
    supported_candidate_fractions: list[float] = []
    pairwise_reliabilities: list[float] = []

    for record in pairwise_ranked_predictions_df.to_dict(orient="records"):
        source_event_id = str(record.get("source_event_id"))
        old_rank = original_lookup.get(source_event_id, [])
        new_rank = _normalize_item_list(record.get("pred_ranked_item_ids"))
        if not new_rank:
            continue

        total += 1
        if new_rank != old_rank:
            changed += 1

        old_positions = {item_id: idx + 1 for idx, item_id in enumerate(old_rank)}
        shifts = [abs((idx + 1) - old_positions.get(item_id, idx + 1)) for idx, item_id in enumerate(new_rank)]
        average_shift_accumulator.append(float(np.mean(shifts)) if shifts else 0.0)
        uncertainty_coverages.append(float(record.get("uncertainty_coverage_rate", np.nan)))
        pairwise_coverages.append(float(record.get("pairwise_pair_coverage_rate", np.nan)))
        supported_candidate_fractions.append(
            float(record.get("pairwise_supported_candidate_fraction", np.nan))
        )
        pairwise_reliabilities.append(float(record.get("pairwise_avg_reliability_weight", np.nan)))

    avg_uncertainty_coverage = (
        float(np.nanmean(uncertainty_coverages)) if uncertainty_coverages else float("nan")
    )
    supported_event_fraction = (
        float(total / total_ranking_events) if total_ranking_events else float("nan")
    )

    return {
        "changed_ranking_fraction": float(changed / total) if total else float("nan"),
        "avg_position_shift": float(np.mean(average_shift_accumulator)) if average_shift_accumulator else float("nan"),
        "avg_uncertainty_coverage_rate": avg_uncertainty_coverage,
        "uncertainty_coverage": avg_uncertainty_coverage,
        "pairwise_pair_coverage_rate": float(np.nanmean(pairwise_coverages)) if pairwise_coverages else float("nan"),
        "pairwise_supported_candidate_fraction": (
            float(np.nanmean(supported_candidate_fractions)) if supported_candidate_fractions else float("nan")
        ),
        "pairwise_avg_reliability_weight": (
            float(np.nanmean(pairwise_reliabilities)) if pairwise_reliabilities else float("nan")
        ),
        "pairwise_supported_event_fraction": supported_event_fraction,
    }

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


LINEAR_VARIANTS = {
    "linear",
    "coverage_aware_linear",
    "topk_gated_linear",
}
STRUCTURED_VARIANT = "nonlinear_structured_risk_rerank"
LOCAL_SWAP_VARIANT = "local_margin_swap_rerank"
STRUCTURED_SWAP_VARIANT = "structured_risk_plus_local_margin_swap_rerank"
SUPPORTED_RERANK_VARIANTS = LINEAR_VARIANTS | {
    STRUCTURED_VARIANT,
    LOCAL_SWAP_VARIANT,
    STRUCTURED_SWAP_VARIANT,
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
        if (
            uncertainty_confidence_col
            and uncertainty_confidence_col in record
            and pd.notna(record[uncertainty_confidence_col])
        ):
            entry["uncertainty_confidence"] = float(record[uncertainty_confidence_col])
        lookup[key] = entry
        uncertainties.append(uncertainty_value)

    fallback_uncertainty = float(np.mean(uncertainties)) if uncertainties else 0.5
    return lookup, fallback_uncertainty


def _position_to_relevance(rank_position: int, num_candidates: int) -> float:
    if num_candidates <= 0:
        return 0.0
    return float((num_candidates - rank_position + 1) / num_candidates)


def _resolve_score_formula_variant(rerank_variant: str) -> str:
    if rerank_variant not in SUPPORTED_RERANK_VARIANTS:
        raise ValueError(f"Unsupported rerank variant: {rerank_variant}")
    if rerank_variant == STRUCTURED_SWAP_VARIANT:
        return STRUCTURED_VARIANT
    if rerank_variant == LOCAL_SWAP_VARIANT:
        return "baseline_relevance_only"
    return rerank_variant


def build_ranker_rows(
    ranking_predictions_df: pd.DataFrame,
    uncertainty_df: pd.DataFrame,
    *,
    lambda_penalty: float,
    topk: int = 10,
    rerank_variant: str = "linear",
    gate_topk: int | None = None,
    tau: float = 0.35,
    gamma: float = 2.0,
    alpha: float = 1.0,
    beta: float = 0.7,
    delta: float = 0.5,
    coverage_fallback_scale: float = 0.5,
    eta: float = 0.02,
    m_rel: float = 0.05,
    m_unc: float = 0.15,
    swap_a: float = 1.5,
    swap_b: float = 1.0,
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
    score_formula_variant = _resolve_score_formula_variant(rerank_variant)

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
        matched_uncertainties: list[float] = []

        for item_id in candidate_ids:
            uncertainty_key = (str(record.get(user_col)), str(item_id))
            uncertainty_entry = lookup.get(uncertainty_key)
            if uncertainty_entry is not None:
                matched_uncertainties.append(float(uncertainty_entry["uncertainty"]))

        event_coverage_rate = float(len(matched_uncertainties) / num_candidates) if num_candidates else 0.0
        event_mean_uncertainty = (
            float(np.mean(matched_uncertainties))
            if matched_uncertainties
            else fallback_uncertainty
        )
        effective_gate_topk = int(max(1, min(gate_topk or topk, num_candidates))) if num_candidates else 0

        for item_id in candidate_ids:
            original_rank = int(rank_lookup.get(item_id, num_candidates + 1))
            relevance_score = _position_to_relevance(original_rank, num_candidates)
            uncertainty_key = (str(record.get(user_col)), str(item_id))
            uncertainty_entry = lookup.get(uncertainty_key)
            if uncertainty_entry is None:
                observed_uncertainty = event_mean_uncertainty
                uncertainty_confidence = float("nan")
                matched_uncertainty = False
            else:
                observed_uncertainty = float(uncertainty_entry["uncertainty"])
                uncertainty_confidence = float(uncertainty_entry.get("uncertainty_confidence", np.nan))
                matched_uncertainty = True

            effective_uncertainty = observed_uncertainty
            coverage_factor = 1.0
            coverage_weight = 1.0
            is_topk_band = original_rank <= effective_gate_topk if effective_gate_topk > 0 else False
            gate_weight = 1.0
            gate_value = 1.0
            position_boost = 1.0
            risk_weight = 1.0
            u_core = 0.0
            u_nonlinear = effective_uncertainty
            protection_bonus = 0.0

            if score_formula_variant == "coverage_aware_linear":
                effective_uncertainty = observed_uncertainty if matched_uncertainty else event_mean_uncertainty
                coverage_weight = event_coverage_rate
                effective_lambda_penalty = float(lambda_penalty) * coverage_weight
                final_score = relevance_score - effective_lambda_penalty * effective_uncertainty
            elif score_formula_variant == "topk_gated_linear":
                gate_weight = 1.0 if is_topk_band else 0.0
                effective_lambda_penalty = float(lambda_penalty) * gate_weight
                final_score = relevance_score - effective_lambda_penalty * effective_uncertainty
            elif score_formula_variant == STRUCTURED_VARIANT:
                coverage_factor = 1.0 if matched_uncertainty else float(coverage_fallback_scale)
                gate_value = (
                    float(np.exp(-float(beta) * max(original_rank - 1, 0)))
                    if original_rank <= effective_gate_topk
                    else 0.0
                )
                position_boost = (
                    1.0 + float(delta) * max(effective_gate_topk - original_rank, 0) / effective_gate_topk
                    if effective_gate_topk > 0
                    else 1.0
                )
                u_core = max(effective_uncertainty - float(tau), 0.0)
                u_nonlinear = (u_core ** float(gamma)) * (1.0 + float(alpha) * effective_uncertainty)
                risk_weight = gate_value * coverage_factor * position_boost
                protection_bonus = float(eta) * relevance_score * (1.0 - effective_uncertainty)
                if not matched_uncertainty:
                    protection_bonus *= float(coverage_fallback_scale)
                effective_lambda_penalty = float(lambda_penalty) * risk_weight
                final_score = (
                    relevance_score
                    - float(lambda_penalty) * risk_weight * u_nonlinear
                    + protection_bonus
                )
            elif score_formula_variant == "baseline_relevance_only":
                effective_lambda_penalty = 0.0
                final_score = relevance_score
            elif score_formula_variant == "linear":
                effective_lambda_penalty = float(lambda_penalty)
                final_score = relevance_score - effective_lambda_penalty * effective_uncertainty
            else:
                raise ValueError(f"Unsupported score formula variant: {score_formula_variant}")

            comparison_uncertainty = (
                effective_uncertainty if matched_uncertainty else effective_uncertainty * float(coverage_fallback_scale)
            )

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
                    "rerank_variant": rerank_variant,
                    "score_formula_variant": score_formula_variant,
                    "gate_topk": int(effective_gate_topk) if effective_gate_topk > 0 else np.nan,
                    "is_topk_band": is_topk_band,
                    "relevance_score": relevance_score,
                    "uncertainty": observed_uncertainty,
                    "effective_uncertainty": effective_uncertainty,
                    "comparison_uncertainty": comparison_uncertainty,
                    "uncertainty_confidence": uncertainty_confidence,
                    "matched_uncertainty": matched_uncertainty,
                    "event_uncertainty_coverage_rate": event_coverage_rate,
                    "event_mean_uncertainty": event_mean_uncertainty,
                    "coverage_weight": coverage_weight,
                    "coverage_factor": coverage_factor,
                    "gate_weight": gate_weight,
                    "gate_value": gate_value,
                    "position_boost": position_boost,
                    "risk_weight": risk_weight,
                    "u_core": u_core,
                    "u_nonlinear": u_nonlinear,
                    "protection_bonus": protection_bonus,
                    "effective_lambda_penalty": effective_lambda_penalty,
                    "final_score": final_score,
                    "uncertainty_source": uncertainty_source,
                    "lambda_penalty": float(lambda_penalty),
                    "lambda": float(lambda_penalty),
                    "tau": float(tau),
                    "gamma": float(gamma),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "delta": float(delta),
                    "coverage_fallback_scale": float(coverage_fallback_scale),
                    "eta": float(eta),
                    "m_rel": float(m_rel),
                    "m_unc": float(m_unc),
                    "swap_a": float(swap_a),
                    "swap_b": float(swap_b),
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
    out["local_swap_applied"] = False
    out["local_swap_iterations"] = 0
    out["swap_gain"] = 0.0
    return out


def apply_local_margin_swaps(
    ranked_df: pd.DataFrame,
    *,
    rerank_variant: str,
    m_rel: float,
    m_unc: float,
    swap_a: float,
    swap_b: float,
    max_iterations: int = 2,
    group_col: str = "source_event_id",
    rank_col: str = "rerank_rank",
) -> pd.DataFrame:
    if rerank_variant not in {LOCAL_SWAP_VARIANT, STRUCTURED_SWAP_VARIANT}:
        return ranked_df

    output_groups: list[pd.DataFrame] = []
    for _, event_df in ranked_df.groupby(group_col, dropna=False):
        event_rows = event_df.sort_values(rank_col).copy().reset_index(drop=True)
        swap_applied = False
        total_swap_gain = 0.0
        completed_iterations = 0

        for iteration_idx in range(max_iterations):
            changed_in_pass = False
            for idx in range(len(event_rows) - 1):
                left = event_rows.iloc[idx]
                right = event_rows.iloc[idx + 1]
                num_candidates = max(int(left.get("num_candidates", len(event_rows))), 1)
                rel_gap = float(left["relevance_score"]) - float(right["relevance_score"])
                normalized_rel_gap = rel_gap / float(num_candidates)
                unc_gap = float(left["comparison_uncertainty"]) - float(right["comparison_uncertainty"])
                swap_gain = float(swap_a) * unc_gap - float(swap_b) * normalized_rel_gap

                if normalized_rel_gap < float(m_rel) and unc_gap > float(m_unc) and swap_gain > 0.0:
                    event_rows.iloc[[idx, idx + 1]] = event_rows.iloc[[idx + 1, idx]].to_numpy()
                    swap_applied = True
                    changed_in_pass = True
                    total_swap_gain += swap_gain

            completed_iterations = iteration_idx + 1
            if not changed_in_pass:
                break

        event_rows[rank_col] = np.arange(1, len(event_rows) + 1)
        event_rows["local_swap_applied"] = swap_applied
        event_rows["local_swap_iterations"] = completed_iterations
        event_rows["swap_gain"] = total_swap_gain
        output_groups.append(event_rows)

    return pd.concat(output_groups, ignore_index=True)


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
                "effective_uncertainty": float(row["effective_uncertainty"]),
                "risk_weight": float(row["risk_weight"]),
                "protection_bonus": float(row["protection_bonus"]),
                "effective_lambda_penalty": float(row["effective_lambda_penalty"]),
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

        first_row = event_df.iloc[0]
        rows.append(
            {
                "user_id": first_row["user_id"],
                "source_event_id": source_event_id,
                "positive_item_id": first_row["positive_item_id"],
                "split_name": first_row["split_name"],
                "timestamp": first_row["timestamp"],
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
                "uncertainty_source": first_row["uncertainty_source"],
                "lambda_penalty": float(first_row["lambda_penalty"]),
                "lambda": float(first_row["lambda"]),
                "rerank_variant": first_row.get("rerank_variant", "linear"),
                "gate_topk": first_row.get("gate_topk", np.nan),
                "tau": float(first_row["tau"]),
                "gamma": float(first_row["gamma"]),
                "alpha": float(first_row["alpha"]),
                "beta": float(first_row["beta"]),
                "delta": float(first_row["delta"]),
                "coverage_fallback_scale": float(first_row["coverage_fallback_scale"]),
                "eta": float(first_row["eta"]),
                "m_rel": float(first_row["m_rel"]),
                "m_unc": float(first_row["m_unc"]),
                "swap_a": float(first_row["swap_a"]),
                "swap_b": float(first_row["swap_b"]),
                "uncertainty_coverage_rate": float(event_df["matched_uncertainty"].mean()),
                "event_uncertainty_coverage_rate": float(event_df["event_uncertainty_coverage_rate"].mean()),
                "local_swap_applied": bool(event_df["local_swap_applied"].any()),
                "local_swap_iterations": int(event_df["local_swap_iterations"].max()),
                "swap_gain": float(event_df["swap_gain"].max()),
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
    local_swap_values = []

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
        local_swap_values.append(1.0 if bool(record.get("local_swap_applied", False)) else 0.0)

    avg_coverage = float(np.nanmean(coverage_values)) if coverage_values else float("nan")
    return {
        "changed_ranking_fraction": float(changed / total) if total else float("nan"),
        "avg_position_shift": float(np.mean(average_shift_accumulator)) if average_shift_accumulator else float("nan"),
        "avg_uncertainty_coverage_rate": avg_coverage,
        "uncertainty_coverage": avg_coverage,
        "local_swap_event_fraction": float(np.mean(local_swap_values)) if local_swap_values else 0.0,
    }

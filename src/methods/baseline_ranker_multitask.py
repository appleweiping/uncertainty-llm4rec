from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from main_rank_rerank import build_result_row, infer_domain_name, infer_model_name, load_jsonl, save_jsonl, save_table
from src.methods.uncertainty_pairwise_aggregator import (
    aggregate_pairwise_preferences,
    build_pairwise_preference_rows,
    build_pairwise_ranked_predictions,
    rank_pairwise_candidates,
    summarize_pairwise_rank_effect,
)
from src.methods.uncertainty_ranker import summarize_rerank_effect
from src.utils.paths import ensure_exp_dirs


ESTIMATOR_PREFERENCE = {
    "verbalized_calibrated": 0,
    "fused": 1,
    "verbalized_raw": 2,
    "consistency": 3,
    "none": 4,
    "direct_reference": 5,
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


def _preferred_estimator_rank(value: Any) -> int:
    return ESTIMATOR_PREFERENCE.get(str(value).strip().lower(), 99)


def _ensure_compare_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "method_family" not in out.columns:
        out["method_family"] = ""
    if "method_variant" not in out.columns:
        out["method_variant"] = ""
    if "notes" not in out.columns:
        out["notes"] = ""
    if "uncertainty_source" not in out.columns:
        out["uncertainty_source"] = ""
    if "estimator" not in out.columns:
        out["estimator"] = ""
    if "evaluation_scope" not in out.columns:
        out["evaluation_scope"] = ""
    if "is_same_task_baseline" not in out.columns:
        out["is_same_task_baseline"] = False
    if "is_current_best_family" not in out.columns:
        out["is_current_best_family"] = False
    return out


def _decorate_compare_row(
    row_df: pd.DataFrame,
    *,
    is_same_task_baseline: bool,
    is_current_best_family: bool,
    notes: str | None = None,
) -> pd.DataFrame:
    out = _ensure_compare_columns(row_df)
    out["is_same_task_baseline"] = bool(is_same_task_baseline)
    out["is_current_best_family"] = bool(is_current_best_family)
    if notes:
        out["notes"] = notes
    return out


def select_best_existing_row(
    compare_df: pd.DataFrame,
    *,
    task: str,
    evaluation_scope: str,
    method_family: str | None = None,
    method_variant: str | None = None,
) -> pd.DataFrame:
    filtered_df = compare_df[
        (compare_df["task"].astype(str) == task)
        & (compare_df["evaluation_scope"].astype(str) == evaluation_scope)
    ].copy()
    if method_family is not None:
        filtered_df = filtered_df[filtered_df["method_family"].astype(str) == method_family].copy()
    if method_variant is not None:
        filtered_df = filtered_df[filtered_df["method_variant"].astype(str) == method_variant].copy()
    if filtered_df.empty:
        raise ValueError(
            f"Cannot find day3 compare row for task={task}, evaluation_scope={evaluation_scope}, "
            f"method_family={method_family}, method_variant={method_variant}."
        )

    for column in ["NDCG@10", "MRR", "changed_ranking_fraction"]:
        if column not in filtered_df.columns:
            filtered_df[column] = np.nan
        filtered_df[column] = pd.to_numeric(filtered_df[column], errors="coerce")

    filtered_df["_estimator_rank"] = filtered_df.get("estimator", "").map(_preferred_estimator_rank)
    filtered_df = filtered_df.sort_values(
        by=["NDCG@10", "MRR", "changed_ranking_fraction", "_estimator_rank", "method"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    return filtered_df.head(1).drop(columns=["_estimator_rank"], errors="ignore").reset_index(drop=True)


def build_pointwise_baseline_compare_rows(
    *,
    output_root: str | Path,
    pointwise_exp_name: str,
) -> pd.DataFrame:
    pointwise_paths = ensure_exp_dirs(pointwise_exp_name, output_root)
    rerank_results_path = pointwise_paths.tables_dir / "rerank_results.csv"
    diagnostic_path = pointwise_paths.tables_dir / "diagnostic_metrics.csv"
    estimator_compare_path = pointwise_paths.tables_dir / "estimator_comparison.csv"
    calibrated_path = pointwise_paths.calibrated_dir / "test_calibrated.jsonl"

    if not rerank_results_path.exists():
        raise FileNotFoundError(f"Pointwise rerank results not found: {rerank_results_path}")
    if not diagnostic_path.exists():
        raise FileNotFoundError(f"Pointwise diagnostic metrics not found: {diagnostic_path}")

    rerank_results_df = pd.read_csv(rerank_results_path)
    diagnostic_row = pd.read_csv(diagnostic_path).iloc[0].to_dict()
    calibrated_metrics_row = diagnostic_row
    if estimator_compare_path.exists():
        estimator_compare_df = pd.read_csv(estimator_compare_path)
        calibrated_rows = estimator_compare_df[
            estimator_compare_df["estimator"].astype(str).str.strip().str.lower() == "verbalized_calibrated"
        ].copy()
        if not calibrated_rows.empty:
            calibrated_metrics_row = calibrated_rows.iloc[0].to_dict()
    calibrated_df = load_jsonl(calibrated_path) if calibrated_path.exists() else pd.DataFrame()

    parse_success_rate = 1.0
    avg_latency = float("nan")
    if not calibrated_df.empty:
        if "response_latency" in calibrated_df.columns:
            avg_latency = float(pd.to_numeric(calibrated_df["response_latency"], errors="coerce").mean())
        parse_success_rate = 1.0

    domain = infer_domain_name(pointwise_exp_name)
    model = infer_model_name(pointwise_exp_name)

    rows: list[dict[str, Any]] = []
    for _, rerank_row in rerank_results_df.iterrows():
        method_name = str(rerank_row.get("method", "")).strip().lower()
        is_baseline = method_name == "baseline"
        if is_baseline:
            method = "pointwise_calibrated_confidence_baseline"
            method_family = "pointwise_same_task_baseline"
            method_variant = "calibrated_confidence_baseline"
            estimator = "none"
            uncertainty_source = "none"
            notes = "old pointwise rerank baseline using calibrated confidence only"
        else:
            method = "pointwise_uncertainty_aware_rerank"
            method_family = "pointwise_uncertainty_aware_family"
            method_variant = "uncertainty_aware_rerank"
            estimator = "verbalized_calibrated"
            uncertainty_source = "verbalized_calibrated"
            notes = "old pointwise uncertainty-aware rerank under the same calibrated input"

        rows.append(
            {
                "domain": domain,
                "model": model,
                "task": "pointwise_yesno",
                "estimator": estimator,
                "method_family": method_family,
                "method_variant": method_variant,
                "evaluation_scope": "full_pointwise_set",
                "sample_count": int(rerank_row.get("num_samples", diagnostic_row.get("num_samples", 0))),
                "samples": int(rerank_row.get("num_samples", diagnostic_row.get("num_samples", 0))),
                "HR@10": pd.to_numeric(rerank_row.get("HR@10"), errors="coerce"),
                "NDCG@10": pd.to_numeric(rerank_row.get("NDCG@10"), errors="coerce"),
                "MRR": pd.to_numeric(rerank_row.get("MRR@10", rerank_row.get("MRR")), errors="coerce"),
                "pairwise_accuracy": float("nan"),
                "pairwise_supported_event_fraction": float("nan"),
                "pairwise_pair_coverage_rate": float("nan"),
                "pairwise_supported_candidate_fraction": float("nan"),
                "coverage": float("nan"),
                "head_exposure": pd.to_numeric(rerank_row.get("head_exposure_ratio@10"), errors="coerce"),
                "longtail_coverage": pd.to_numeric(rerank_row.get("long_tail_coverage@10"), errors="coerce"),
                "ECE": pd.to_numeric(
                    calibrated_metrics_row.get("calib_ece", calibrated_metrics_row.get("ece")),
                    errors="coerce",
                ),
                "Brier": pd.to_numeric(
                    calibrated_metrics_row.get("calib_brier_score", calibrated_metrics_row.get("brier_score")),
                    errors="coerce",
                ),
                "AUROC": pd.to_numeric(
                    calibrated_metrics_row.get("calib_auroc", calibrated_metrics_row.get("auroc")),
                    errors="coerce",
                ),
                "avg_confidence": pd.to_numeric(
                    calibrated_metrics_row.get("calib_avg_confidence", calibrated_metrics_row.get("avg_confidence")),
                    errors="coerce",
                ),
                "accuracy": pd.to_numeric(
                    calibrated_metrics_row.get("calib_accuracy", calibrated_metrics_row.get("accuracy")),
                    errors="coerce",
                ),
                "correctness_correlation": float("nan"),
                "parse_success_rate": parse_success_rate,
                "avg_latency": avg_latency,
                "uncertainty_coverage": 1.0 if not is_baseline else float("nan"),
                "uncertainty_source": uncertainty_source,
                "notes": notes,
                "method": method,
                "avg_candidates": float("nan"),
                "coverage@10": float("nan"),
                "head_exposure_ratio@10": pd.to_numeric(rerank_row.get("head_exposure_ratio@10"), errors="coerce"),
                "longtail_coverage@10": pd.to_numeric(rerank_row.get("long_tail_coverage@10"), errors="coerce"),
                "out_of_candidate_rate": float("nan"),
                "head_exposure_ratio": pd.to_numeric(rerank_row.get("head_exposure_ratio@10"), errors="coerce"),
                "changed_ranking_fraction": float("nan"),
                "avg_position_shift": float("nan"),
                "avg_uncertainty_coverage_rate": 1.0 if not is_baseline else float("nan"),
                "lambda_penalty": pd.to_numeric(rerank_row.get("lambda_penalty"), errors="coerce"),
                "lambda": pd.to_numeric(rerank_row.get("lambda_penalty"), errors="coerce"),
            }
        )

    pointwise_df = pd.DataFrame(rows)
    pointwise_df = _decorate_compare_row(
        pointwise_df,
        is_same_task_baseline=False,
        is_current_best_family=False,
    )
    pointwise_df.loc[pointwise_df["method_variant"] == "calibrated_confidence_baseline", "is_same_task_baseline"] = True
    return pointwise_df


def _combine_pairwise_with_fallback(
    pairwise_ranked_predictions: pd.DataFrame,
    ranking_df: pd.DataFrame,
    *,
    fallback_uncertainty_source: str,
) -> pd.DataFrame:
    support_event_ids = set(pairwise_ranked_predictions["source_event_id"].astype(str).tolist())
    fallback_records: list[dict[str, Any]] = []
    for record in ranking_df.to_dict(orient="records"):
        if str(record.get("source_event_id")) in support_event_ids:
            continue
        fallback_record = dict(record)
        fallback_record["original_pred_ranked_item_ids"] = _normalize_item_list(record.get("pred_ranked_item_ids"))
        fallback_record["pairwise_exp_pair_count"] = 0
        fallback_record["pairwise_pair_coverage_rate"] = 0.0
        fallback_record["pairwise_supported_candidate_fraction"] = 0.0
        fallback_record["pairwise_avg_reliability_weight"] = float("nan")
        fallback_record["pairwise_accuracy"] = float("nan")
        fallback_record["aggregation_variant"] = "direct_ranking_fallback"
        fallback_record["prior_weight"] = 0.0
        fallback_record["loss_weight"] = 0.0
        fallback_record["score_scale"] = 0.0
        fallback_record["uncertainty_source"] = fallback_uncertainty_source
        fallback_record["uncertainty_coverage_rate"] = float("nan")
        fallback_record["total_ranking_events"] = len(ranking_df)
        fallback_records.append(fallback_record)

    return pd.concat(
        [pairwise_ranked_predictions, pd.DataFrame(fallback_records)],
        ignore_index=True,
        sort=False,
    )


def build_pairwise_plain_baseline_rows(
    *,
    output_root: str | Path,
    pointwise_exp_name: str,
    ranking_exp_name: str,
    pairwise_exp_name: str,
    pairwise_baseline_exp_name: str,
    topk: int,
    prior_weight: float,
    loss_weight: float,
    score_scale: float,
) -> pd.DataFrame:
    pointwise_paths = ensure_exp_dirs(pointwise_exp_name, output_root)
    ranking_paths = ensure_exp_dirs(ranking_exp_name, output_root)
    pairwise_paths = ensure_exp_dirs(pairwise_exp_name, output_root)
    baseline_paths = ensure_exp_dirs(pairwise_baseline_exp_name, output_root)

    calibrated_path = pointwise_paths.calibrated_dir / "test_calibrated.jsonl"
    ranking_path = ranking_paths.predictions_dir / "rank_predictions.jsonl"
    pairwise_path = pairwise_paths.predictions_dir / "pairwise_predictions.jsonl"
    if not calibrated_path.exists():
        raise FileNotFoundError(f"Pointwise calibrated file not found: {calibrated_path}")
    if not ranking_path.exists():
        raise FileNotFoundError(f"Ranking prediction file not found: {ranking_path}")
    if not pairwise_path.exists():
        raise FileNotFoundError(f"Pairwise prediction file not found: {pairwise_path}")

    uncertainty_df = load_jsonl(calibrated_path)
    ranking_df = load_jsonl(ranking_path)
    pairwise_df = load_jsonl(pairwise_path)

    pairwise_rows_df = build_pairwise_preference_rows(
        pairwise_predictions_df=pairwise_df,
        ranking_predictions_df=ranking_df,
        uncertainty_df=uncertainty_df,
        uncertainty_col="uncertainty",
        uncertainty_confidence_col="calibrated_confidence",
        uncertainty_source="baseline_context_only",
    )
    if pairwise_rows_df.empty:
        raise RuntimeError("Pairwise plain baseline cannot be built because aligned pairwise rows are empty.")

    plain_pairwise_rows_df = pairwise_rows_df.copy()
    plain_pairwise_rows_df["pair_reliability_weight"] = 1.0
    plain_pairwise_rows_df["pair_uncertainty"] = float("nan")
    plain_pairwise_rows_df["uncertainty_source"] = "none"

    aggregated_item_df = aggregate_pairwise_preferences(
        plain_pairwise_rows_df,
        ranking_df,
        aggregation_variant="weighted_win_count",
        prior_weight=prior_weight,
        loss_weight=loss_weight,
        score_scale=score_scale,
    )
    ranked_item_df = rank_pairwise_candidates(aggregated_item_df)
    plain_ranked_predictions = build_pairwise_ranked_predictions(
        ranked_item_df,
        ranking_df,
        topk=topk,
        total_ranking_events=len(ranking_df),
    )
    plain_ranked_predictions["uncertainty_source"] = "none"

    save_table(plain_pairwise_rows_df, baseline_paths.tables_dir / "pairwise_plain_preference_rows.csv")
    save_table(ranked_item_df, baseline_paths.tables_dir / "pairwise_plain_rank_item_scores.csv")
    save_jsonl(plain_ranked_predictions, baseline_paths.reranked_dir / "pairwise_plain_ranked.jsonl")

    support_event_ids = set(plain_ranked_predictions["source_event_id"].astype(str).tolist())
    overlap_ranking_df = ranking_df[ranking_df["source_event_id"].astype(str).isin(support_event_ids)].copy()
    overlap_effect_metrics = summarize_pairwise_rank_effect(
        overlap_ranking_df,
        plain_ranked_predictions,
        total_ranking_events=len(ranking_df),
    )
    overlap_row_df = build_result_row(
        method_name="pairwise_plain_win_count",
        prediction_df=plain_ranked_predictions,
        topk=topk,
        uncertainty_source="none",
        extra_metrics=overlap_effect_metrics,
    )

    expanded_predictions = _combine_pairwise_with_fallback(
        plain_ranked_predictions,
        ranking_df,
        fallback_uncertainty_source="none",
    )
    save_jsonl(expanded_predictions, baseline_paths.reranked_dir / "pairwise_plain_ranked_expanded.jsonl")

    expanded_effect_metrics = summarize_rerank_effect(ranking_df, expanded_predictions)
    for metric_name in [
        "pairwise_supported_event_fraction",
        "pairwise_pair_coverage_rate",
        "pairwise_supported_candidate_fraction",
        "pairwise_avg_reliability_weight",
    ]:
        expanded_effect_metrics[metric_name] = overlap_effect_metrics.get(metric_name, float("nan"))
    expanded_row_df = build_result_row(
        method_name="pairwise_plain_win_count_expanded",
        prediction_df=expanded_predictions,
        topk=topk,
        uncertainty_source="none",
        extra_metrics=expanded_effect_metrics,
    )

    domain = infer_domain_name(ranking_exp_name)
    model = infer_model_name(ranking_exp_name)
    pairwise_accuracy = float(
        (
            pairwise_rows_df["preferred_item_true"].astype(str)
            == pairwise_rows_df["preferred_item_pred"].astype(str)
        ).mean()
    )

    rows: list[pd.DataFrame] = []
    for row_df, method_variant, evaluation_scope, notes in [
        (
            overlap_row_df,
            "plain_win_count_overlap",
            "pairwise_event_overlap_subset",
            "pairwise plain aggregation baseline on pairwise-supported overlap subset",
        ),
        (
            expanded_row_df,
            "plain_win_count_expanded",
            "expanded_with_direct_fallback",
            "pairwise plain aggregation baseline with direct-ranking fallback on unsupported events",
        ),
    ]:
        row_df = _ensure_compare_columns(row_df)
        row_df["domain"] = domain
        row_df["model"] = model
        row_df["task"] = "pairwise_to_rank"
        row_df["estimator"] = "none"
        row_df["method_family"] = "pairwise_plain_aggregation_baseline"
        row_df["method_variant"] = method_variant
        row_df["evaluation_scope"] = evaluation_scope
        row_df["uncertainty_source"] = "none"
        row_df["notes"] = notes
        row_df["head_exposure"] = row_df.get("head_exposure_ratio")
        row_df["AUROC"] = float("nan")
        row_df["ECE"] = float("nan")
        row_df["Brier"] = float("nan")
        row_df["accuracy"] = float("nan")
        row_df["correctness_correlation"] = float("nan")
        row_df["pairwise_accuracy"] = pairwise_accuracy
        row_df["prior_weight"] = float(prior_weight)
        row_df["loss_weight"] = float(loss_weight)
        row_df["score_scale"] = float(score_scale)
        row_df["pairwise_aggregation_variant"] = "plain_win_count"
        row_df["is_same_task_baseline"] = True
        row_df["is_current_best_family"] = False
        rows.append(row_df)

    plain_results_df = pd.concat(rows, ignore_index=True, sort=False)
    save_table(plain_results_df, baseline_paths.tables_dir / "rerank_results.csv")
    return plain_results_df

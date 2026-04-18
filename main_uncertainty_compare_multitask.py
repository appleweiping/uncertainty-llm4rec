from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from main_rank_rerank import (
    build_result_row,
    infer_domain_name,
    infer_model_name,
    load_jsonl,
    save_table,
)
from src.eval.calibration_metrics import compute_calibration_metrics, ensure_binary_columns
from src.methods.uncertainty_pairwise_aggregator import (
    aggregate_pairwise_preferences,
    build_pairwise_preference_rows,
    build_pairwise_ranked_predictions,
    rank_pairwise_candidates,
    summarize_pairwise_rank_effect,
)
from src.methods.uncertainty_ranker import (
    apply_local_margin_swaps,
    build_ranker_rows,
    build_reranked_predictions,
    rank_candidates_by_score,
    summarize_rerank_effect,
)
from src.uncertainty.estimators import (
    ensure_estimator_columns,
    get_available_estimators,
    merge_consistency_outputs,
)
from src.utils.paths import ensure_exp_dirs
from src.utils.reproducibility import set_global_seed


RANKING_VARIANT_LABELS = {
    "nonlinear_structured_risk_rerank": "structured_risk_family",
    "local_margin_swap_rerank": "local_margin_swap_family",
    "structured_risk_plus_local_margin_swap_rerank": "structured_risk_plus_local_swap_family",
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


def _safe_corr(series_a: pd.Series, series_b: pd.Series) -> float:
    df = pd.DataFrame({"a": series_a, "b": series_b}).dropna()
    if len(df) < 2:
        return float("nan")
    return float(df["a"].corr(df["b"]))


def _load_pointwise_base_df(
    *,
    output_root: str | Path,
    pointwise_exp_name: str,
    fused_alpha: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, str | float]]]:
    pointwise_paths = ensure_exp_dirs(pointwise_exp_name, output_root)
    calibrated_path = pointwise_paths.calibrated_dir / "test_calibrated.jsonl"
    consistency_path = pointwise_paths.root / "self_consistency" / "test_self_consistency.jsonl"

    if not calibrated_path.exists():
        raise FileNotFoundError(f"Pointwise calibrated file not found: {calibrated_path}")

    df = load_jsonl(calibrated_path)
    df = ensure_binary_columns(df)

    if consistency_path.exists():
        consistency_df = load_jsonl(consistency_path)
        df = merge_consistency_outputs(df, consistency_df)

    df = ensure_estimator_columns(df, fused_alpha=fused_alpha)
    estimators = get_available_estimators(df, fused_alpha=fused_alpha)
    if not estimators:
        raise ValueError("No uncertainty estimators are available after loading pointwise artifacts.")
    return df, estimators


def _build_estimator_uncertainty_df(
    pointwise_df: pd.DataFrame,
    *,
    confidence_col: str,
    uncertainty_col: str,
) -> pd.DataFrame:
    required_cols = ["user_id", "candidate_item_id", confidence_col, uncertainty_col]
    missing_cols = [col for col in required_cols if col not in pointwise_df.columns]
    if missing_cols:
        raise ValueError(f"Missing estimator columns in pointwise dataframe: {missing_cols}")

    uncertainty_df = (
        pointwise_df[required_cols]
        .dropna(subset=["user_id", "candidate_item_id", uncertainty_col])
        .drop_duplicates(subset=["user_id", "candidate_item_id"])
        .copy()
        .rename(
            columns={
                confidence_col: "estimator_confidence",
                uncertainty_col: "estimator_uncertainty",
            }
        )
    )
    return uncertainty_df


def _load_ranking_family_params(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Ranking family meta file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    return {
        "nonlinear_structured_risk_rerank": meta["structured_best"],
        "local_margin_swap_rerank": meta["local_best"],
        "structured_risk_plus_local_margin_swap_rerank": meta["combo_best"],
    }


def _attach_standard_columns(
    row_df: pd.DataFrame,
    *,
    domain: str,
    model: str,
    task: str,
    estimator: str,
    method_family: str,
    method_variant: str,
    evaluation_scope: str,
    uncertainty_source: str,
    notes: str = "",
) -> pd.DataFrame:
    out = row_df.copy()
    out.insert(0, "model", model)
    out.insert(0, "domain", domain)
    out.insert(2, "task", task)
    out.insert(3, "estimator", estimator)
    out.insert(4, "method_family", method_family)
    out.insert(5, "method_variant", method_variant)
    out.insert(6, "evaluation_scope", evaluation_scope)
    out["uncertainty_source"] = uncertainty_source
    out["notes"] = notes
    return out


def _evaluate_pointwise_rows(
    pointwise_df: pd.DataFrame,
    estimators: dict[str, dict[str, str | float]],
    *,
    domain: str,
    model: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for estimator_name, cols in estimators.items():
        confidence_col = str(cols["confidence_col"])
        uncertainty_col = str(cols["uncertainty_col"])
        eval_df = pointwise_df[
            pointwise_df[confidence_col].notna() & pointwise_df[uncertainty_col].notna()
        ].copy()
        metrics = compute_calibration_metrics(
            eval_df,
            confidence_col=confidence_col,
            target_col="is_correct",
        )
        rows.append(
            {
                "domain": domain,
                "model": model,
                "task": "pointwise_yesno",
                "estimator": estimator_name,
                "method_family": "pointwise_diagnosis",
                "method_variant": "calibration_quality",
                "evaluation_scope": "full_pointwise_set",
                "sample_count": int(len(eval_df)),
                "samples": int(len(eval_df)),
                "HR@10": float("nan"),
                "NDCG@10": float("nan"),
                "MRR": float("nan"),
                "pairwise_accuracy": float("nan"),
                "pairwise_supported_event_fraction": float("nan"),
                "pairwise_pair_coverage_rate": float("nan"),
                "pairwise_supported_candidate_fraction": float("nan"),
                "coverage": float("nan"),
                "head_exposure": float("nan"),
                "longtail_coverage": float("nan"),
                "ECE": metrics.get("ece"),
                "Brier": metrics.get("brier_score"),
                "AUROC": metrics.get("auroc"),
                "avg_confidence": metrics.get("avg_confidence"),
                "accuracy": metrics.get("accuracy"),
                "correctness_correlation": _safe_corr(
                    eval_df[confidence_col],
                    eval_df["is_correct"],
                ),
                "parse_success_rate": float(eval_df.get("parse_success", False).fillna(False).astype(bool).mean())
                if "parse_success" in eval_df.columns
                else float("nan"),
                "avg_latency": float(pd.to_numeric(eval_df.get("response_latency"), errors="coerce").mean())
                if "response_latency" in eval_df.columns
                else float("nan"),
                "uncertainty_coverage": float(eval_df[uncertainty_col].notna().mean()),
                "uncertainty_source": estimator_name,
                "notes": "pointwise diagnostic view",
            }
        )

    return pd.DataFrame(rows)


def _evaluate_ranking_rows(
    ranking_df: pd.DataFrame,
    uncertainty_df: pd.DataFrame,
    *,
    estimator_name: str,
    ranking_family_params: dict[str, dict[str, float]],
    domain: str,
    model: str,
    topk: int,
) -> list[pd.DataFrame]:
    result_frames: list[pd.DataFrame] = []

    for rerank_variant, params in ranking_family_params.items():
        scored_rows = build_ranker_rows(
            ranking_predictions_df=ranking_df,
            uncertainty_df=uncertainty_df,
            lambda_penalty=float(params["lambda_penalty"]),
            topk=topk,
            rerank_variant=rerank_variant,
            gate_topk=int(params["gate_topk"]),
            tau=float(params["tau"]),
            gamma=float(params["gamma"]),
            alpha=float(params["alpha"]),
            beta=float(params["beta"]),
            delta=float(params["delta"]),
            coverage_fallback_scale=float(params["coverage_fallback_scale"]),
            eta=float(params["eta"]),
            m_rel=float(params["m_rel"]),
            m_unc=float(params["m_unc"]),
            swap_a=float(params["swap_a"]),
            swap_b=float(params["swap_b"]),
            uncertainty_col="estimator_uncertainty",
            uncertainty_confidence_col="estimator_confidence",
            uncertainty_source=estimator_name,
        )
        ranked_rows = rank_candidates_by_score(scored_rows)
        ranked_rows = apply_local_margin_swaps(
            ranked_rows,
            rerank_variant=rerank_variant,
            m_rel=float(params["m_rel"]),
            m_unc=float(params["m_unc"]),
            swap_a=float(params["swap_a"]),
            swap_b=float(params["swap_b"]),
            max_iterations=2,
        )
        reranked_predictions = build_reranked_predictions(ranked_rows, ranking_df, topk=topk)
        effect_metrics = summarize_rerank_effect(ranking_df, reranked_predictions)
        row_df = build_result_row(
            method_name=f"uncertainty_aware_rank_rerank_{rerank_variant}",
            prediction_df=reranked_predictions,
            topk=topk,
            lambda_penalty=float(params["lambda_penalty"]),
            uncertainty_source=estimator_name,
            rerank_variant=rerank_variant,
            gate_topk=int(params["gate_topk"]),
            tau=float(params["tau"]),
            gamma=float(params["gamma"]),
            alpha=float(params["alpha"]),
            beta=float(params["beta"]),
            delta=float(params["delta"]),
            coverage_fallback_scale=float(params["coverage_fallback_scale"]),
            eta=float(params["eta"]),
            m_rel=float(params["m_rel"]),
            m_unc=float(params["m_unc"]),
            swap_a=float(params["swap_a"]),
            swap_b=float(params["swap_b"]),
            extra_metrics=effect_metrics,
        )
        row_df["head_exposure"] = row_df.get("head_exposure_ratio")
        row_df["AUROC"] = float("nan")
        row_df["ECE"] = float("nan")
        row_df["Brier"] = float("nan")
        row_df["accuracy"] = float("nan")
        row_df["correctness_correlation"] = float("nan")
        row_df["pairwise_accuracy"] = float("nan")
        row_df["pairwise_supported_event_fraction"] = float("nan")
        row_df["pairwise_pair_coverage_rate"] = float("nan")
        row_df["pairwise_supported_candidate_fraction"] = float("nan")
        row_df = _attach_standard_columns(
            row_df,
            domain=domain,
            model=model,
            task="candidate_ranking",
            estimator=estimator_name,
            method_family=RANKING_VARIANT_LABELS[rerank_variant],
            method_variant=rerank_variant,
            evaluation_scope="full_ranking_set",
            uncertainty_source=estimator_name,
            notes="ranking family compare under unified estimator setting",
        )
        result_frames.append(row_df)

    return result_frames


def _combine_pairwise_with_fallback(
    pairwise_ranked_predictions: pd.DataFrame,
    ranking_df: pd.DataFrame,
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
        fallback_record["uncertainty_source"] = "direct_ranking_fallback"
        fallback_record["uncertainty_coverage_rate"] = float("nan")
        fallback_record["total_ranking_events"] = len(ranking_df)
        fallback_records.append(fallback_record)

    expanded_df = pd.concat(
        [pairwise_ranked_predictions, pd.DataFrame(fallback_records)],
        ignore_index=True,
        sort=False,
    )
    return expanded_df


def _evaluate_pairwise_rows(
    pairwise_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    uncertainty_df: pd.DataFrame,
    *,
    estimator_name: str,
    aggregation_variant: str,
    prior_weight: float,
    loss_weight: float,
    score_scale: float,
    domain: str,
    model: str,
    topk: int,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    result_frames: list[pd.DataFrame] = []
    pairwise_rows_df = build_pairwise_preference_rows(
        pairwise_predictions_df=pairwise_df,
        ranking_predictions_df=ranking_df,
        uncertainty_df=uncertainty_df,
        uncertainty_col="estimator_uncertainty",
        uncertainty_confidence_col="estimator_confidence",
        uncertainty_source=estimator_name,
    )
    aggregated_item_df = aggregate_pairwise_preferences(
        pairwise_rows_df,
        ranking_df,
        aggregation_variant=aggregation_variant,
        prior_weight=prior_weight,
        loss_weight=loss_weight,
        score_scale=score_scale,
    )
    ranked_item_df = rank_pairwise_candidates(aggregated_item_df)
    pairwise_ranked_predictions = build_pairwise_ranked_predictions(
        ranked_item_df,
        ranking_df,
        topk=topk,
        total_ranking_events=len(ranking_df),
    )

    support_event_ids = set(pairwise_ranked_predictions["source_event_id"].astype(str).tolist())
    overlap_ranking_df = ranking_df[ranking_df["source_event_id"].astype(str).isin(support_event_ids)].copy()
    overlap_effect_metrics = summarize_pairwise_rank_effect(
        overlap_ranking_df,
        pairwise_ranked_predictions,
        total_ranking_events=len(ranking_df),
    )
    overlap_row_df = build_result_row(
        method_name=f"pairwise_to_rank_{aggregation_variant}",
        prediction_df=pairwise_ranked_predictions,
        topk=topk,
        uncertainty_source=estimator_name,
        extra_metrics=overlap_effect_metrics,
    )
    overlap_row_df["head_exposure"] = overlap_row_df.get("head_exposure_ratio")
    overlap_row_df["AUROC"] = float("nan")
    overlap_row_df["ECE"] = float("nan")
    overlap_row_df["Brier"] = float("nan")
    overlap_row_df["accuracy"] = float("nan")
    overlap_row_df["correctness_correlation"] = float("nan")
    overlap_row_df["pairwise_accuracy"] = float(pairwise_rows_df["preferred_item_pred"].eq(pairwise_rows_df["preferred_item_true"]).mean()) if not pairwise_rows_df.empty else float("nan")
    overlap_row_df = _attach_standard_columns(
        overlap_row_df,
        domain=domain,
        model=model,
        task="pairwise_to_rank",
        estimator=estimator_name,
        method_family="pairwise_to_rank",
        method_variant=aggregation_variant,
        evaluation_scope="pairwise_event_overlap_subset",
        uncertainty_source=estimator_name,
        notes="pairwise-supported overlap subset",
    )
    overlap_row_df["prior_weight"] = prior_weight
    overlap_row_df["loss_weight"] = loss_weight
    overlap_row_df["score_scale"] = score_scale
    result_frames.append(overlap_row_df)

    expanded_predictions = _combine_pairwise_with_fallback(pairwise_ranked_predictions, ranking_df)
    expanded_effect_metrics = summarize_rerank_effect(ranking_df, expanded_predictions)
    expanded_effect_metrics["pairwise_supported_event_fraction"] = overlap_effect_metrics.get(
        "pairwise_supported_event_fraction",
        float("nan"),
    )
    expanded_effect_metrics["pairwise_pair_coverage_rate"] = overlap_effect_metrics.get(
        "pairwise_pair_coverage_rate",
        float("nan"),
    )
    expanded_effect_metrics["pairwise_supported_candidate_fraction"] = overlap_effect_metrics.get(
        "pairwise_supported_candidate_fraction",
        float("nan"),
    )
    expanded_effect_metrics["pairwise_avg_reliability_weight"] = overlap_effect_metrics.get(
        "pairwise_avg_reliability_weight",
        float("nan"),
    )
    expanded_row_df = build_result_row(
        method_name=f"pairwise_to_rank_{aggregation_variant}_expanded",
        prediction_df=expanded_predictions,
        topk=topk,
        uncertainty_source=estimator_name,
        extra_metrics=expanded_effect_metrics,
    )
    expanded_row_df["head_exposure"] = expanded_row_df.get("head_exposure_ratio")
    expanded_row_df["AUROC"] = float("nan")
    expanded_row_df["ECE"] = float("nan")
    expanded_row_df["Brier"] = float("nan")
    expanded_row_df["accuracy"] = float("nan")
    expanded_row_df["correctness_correlation"] = float("nan")
    expanded_row_df["pairwise_accuracy"] = overlap_row_df["pairwise_accuracy"].iloc[0]
    expanded_row_df = _attach_standard_columns(
        expanded_row_df,
        domain=domain,
        model=model,
        task="pairwise_to_rank",
        estimator=estimator_name,
        method_family="pairwise_to_rank",
        method_variant=f"{aggregation_variant}_expanded",
        evaluation_scope="expanded_with_direct_fallback",
        uncertainty_source=estimator_name,
        notes="pairwise events plus direct-ranking fallback on unsupported events",
    )
    expanded_row_df["prior_weight"] = prior_weight
    expanded_row_df["loss_weight"] = loss_weight
    expanded_row_df["score_scale"] = score_scale
    result_frames.append(expanded_row_df)

    diagnostics_row = {
        "domain": domain,
        "model": model,
        "estimator": estimator_name,
        "supported_event_count": int(len(support_event_ids)),
        "total_ranking_event_count": int(len(ranking_df)),
        "pairwise_supported_event_fraction": float(len(support_event_ids) / len(ranking_df)) if len(ranking_df) else float("nan"),
        "total_pair_count": int(len(pairwise_rows_df)),
        "avg_pairs_per_supported_event": float(len(pairwise_rows_df) / len(support_event_ids)) if support_event_ids else float("nan"),
        "pairwise_pair_coverage_rate": float(pairwise_ranked_predictions["pairwise_pair_coverage_rate"].mean()) if not pairwise_ranked_predictions.empty else float("nan"),
        "avg_pair_reliability_weight": float(pairwise_ranked_predictions["pairwise_avg_reliability_weight"].mean()) if not pairwise_ranked_predictions.empty else float("nan"),
        "pairwise_supported_candidate_fraction": float(pairwise_ranked_predictions["pairwise_supported_candidate_fraction"].mean()) if not pairwise_ranked_predictions.empty else float("nan"),
        "uncertainty_coverage": float(pairwise_ranked_predictions["uncertainty_coverage_rate"].mean()) if not pairwise_ranked_predictions.empty else float("nan"),
        "overlap_NDCG@10": float(overlap_row_df["NDCG@10"].iloc[0]),
        "overlap_MRR": float(overlap_row_df["MRR"].iloc[0]),
        "expanded_NDCG@10": float(expanded_row_df["NDCG@10"].iloc[0]),
        "expanded_MRR": float(expanded_row_df["MRR"].iloc[0]),
        "coverage_expansion_strategy": "direct_ranking_fallback",
    }

    return result_frames, diagnostics_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointwise_exp_name", type=str, default="beauty_qwen")
    parser.add_argument("--ranking_exp_name", type=str, default="beauty_qwen_rank")
    parser.add_argument("--pairwise_exp_name", type=str, default="beauty_qwen_pairwise")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--fused_alpha", type=float, default=0.5)
    parser.add_argument("--pairwise_aggregation_variant", type=str, default="weighted_win_count")
    parser.add_argument("--pairwise_prior_weight", type=float, default=0.2)
    parser.add_argument("--pairwise_loss_weight", type=float, default=1.0)
    parser.add_argument("--pairwise_score_scale", type=float, default=1.0)
    parser.add_argument(
        "--ranking_meta_path",
        type=str,
        default="outputs/summary/week6_rank_rerank_search_meta.json",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    domain = infer_domain_name(args.ranking_exp_name)
    model = infer_model_name(args.ranking_exp_name)

    pointwise_df, estimators = _load_pointwise_base_df(
        output_root=args.output_root,
        pointwise_exp_name=args.pointwise_exp_name,
        fused_alpha=args.fused_alpha,
    )
    ranking_paths = ensure_exp_dirs(args.ranking_exp_name, args.output_root)
    pairwise_paths = ensure_exp_dirs(args.pairwise_exp_name, args.output_root)

    ranking_path = ranking_paths.predictions_dir / "rank_predictions.jsonl"
    pairwise_path = pairwise_paths.predictions_dir / "pairwise_predictions.jsonl"
    if not ranking_path.exists():
        raise FileNotFoundError(f"Ranking prediction file not found: {ranking_path}")
    if not pairwise_path.exists():
        raise FileNotFoundError(f"Pairwise prediction file not found: {pairwise_path}")

    ranking_df = load_jsonl(ranking_path)
    pairwise_df = load_jsonl(pairwise_path)
    ranking_family_params = _load_ranking_family_params(Path(args.ranking_meta_path))

    compare_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []

    pointwise_compare_df = _evaluate_pointwise_rows(
        pointwise_df,
        estimators,
        domain=domain,
        model=model,
    )
    compare_frames.append(pointwise_compare_df)

    direct_ranking_row_df = build_result_row(
        method_name="direct_candidate_ranking",
        prediction_df=ranking_df,
        topk=args.topk,
        extra_metrics={
            "changed_ranking_fraction": 0.0,
            "avg_position_shift": 0.0,
            "avg_uncertainty_coverage_rate": float("nan"),
            "uncertainty_coverage": float("nan"),
            "pairwise_supported_event_fraction": float("nan"),
            "pairwise_pair_coverage_rate": float("nan"),
            "pairwise_supported_candidate_fraction": float("nan"),
        },
    )
    direct_ranking_row_df["head_exposure"] = direct_ranking_row_df.get("head_exposure_ratio")
    direct_ranking_row_df["AUROC"] = float("nan")
    direct_ranking_row_df["ECE"] = float("nan")
    direct_ranking_row_df["Brier"] = float("nan")
    direct_ranking_row_df["accuracy"] = float("nan")
    direct_ranking_row_df["correctness_correlation"] = float("nan")
    direct_ranking_row_df["pairwise_accuracy"] = float("nan")
    direct_ranking_row_df = _attach_standard_columns(
        direct_ranking_row_df,
        domain=domain,
        model=model,
        task="candidate_ranking",
        estimator="direct_reference",
        method_family="direct_candidate_ranking",
        method_variant="direct_candidate_ranking",
        evaluation_scope="full_ranking_set",
        uncertainty_source="none",
        notes="reference ranking output without uncertainty",
    )
    compare_frames.append(direct_ranking_row_df)

    overlap_support_event_ids = set(pairwise_df["source_event_id"].astype(str).tolist())
    direct_overlap_df = ranking_df[ranking_df["source_event_id"].astype(str).isin(overlap_support_event_ids)].copy()
    if not direct_overlap_df.empty:
        direct_overlap_row_df = build_result_row(
            method_name="direct_candidate_ranking_overlap_reference",
            prediction_df=direct_overlap_df,
            topk=args.topk,
            extra_metrics={
                "changed_ranking_fraction": 0.0,
                "avg_position_shift": 0.0,
                "avg_uncertainty_coverage_rate": float("nan"),
                "uncertainty_coverage": float("nan"),
                "pairwise_supported_event_fraction": (
                    float(len(direct_overlap_df) / len(ranking_df)) if len(ranking_df) else float("nan")
                ),
                "pairwise_pair_coverage_rate": float("nan"),
                "pairwise_supported_candidate_fraction": float("nan"),
            },
        )
        direct_overlap_row_df["head_exposure"] = direct_overlap_row_df.get("head_exposure_ratio")
        direct_overlap_row_df["AUROC"] = float("nan")
        direct_overlap_row_df["ECE"] = float("nan")
        direct_overlap_row_df["Brier"] = float("nan")
        direct_overlap_row_df["accuracy"] = float("nan")
        direct_overlap_row_df["correctness_correlation"] = float("nan")
        direct_overlap_row_df["pairwise_accuracy"] = float("nan")
        direct_overlap_row_df = _attach_standard_columns(
            direct_overlap_row_df,
            domain=domain,
            model=model,
            task="pairwise_to_rank",
            estimator="direct_overlap_reference",
            method_family="direct_candidate_ranking",
            method_variant="direct_overlap_reference",
            evaluation_scope="pairwise_event_overlap_subset",
            uncertainty_source="none",
            notes="reference direct ranking on pairwise-supported overlap subset",
        )
        compare_frames.append(direct_overlap_row_df)

    direct_expanded_reference_df = build_result_row(
        method_name="direct_candidate_ranking_expanded_reference",
        prediction_df=ranking_df,
        topk=args.topk,
        extra_metrics={
            "changed_ranking_fraction": 0.0,
            "avg_position_shift": 0.0,
            "avg_uncertainty_coverage_rate": float("nan"),
            "uncertainty_coverage": float("nan"),
            "pairwise_supported_event_fraction": (
                float(len(overlap_support_event_ids) / len(ranking_df)) if len(ranking_df) else float("nan")
            ),
            "pairwise_pair_coverage_rate": float("nan"),
            "pairwise_supported_candidate_fraction": float("nan"),
        },
    )
    direct_expanded_reference_df["head_exposure"] = direct_expanded_reference_df.get("head_exposure_ratio")
    direct_expanded_reference_df["AUROC"] = float("nan")
    direct_expanded_reference_df["ECE"] = float("nan")
    direct_expanded_reference_df["Brier"] = float("nan")
    direct_expanded_reference_df["accuracy"] = float("nan")
    direct_expanded_reference_df["correctness_correlation"] = float("nan")
    direct_expanded_reference_df["pairwise_accuracy"] = float("nan")
    direct_expanded_reference_df = _attach_standard_columns(
        direct_expanded_reference_df,
        domain=domain,
        model=model,
        task="pairwise_to_rank",
        estimator="direct_expanded_reference",
        method_family="direct_candidate_ranking",
        method_variant="direct_expanded_reference",
        evaluation_scope="expanded_with_direct_fallback",
        uncertainty_source="none",
        notes="reference direct ranking on full set before pairwise coverage expansion",
    )
    compare_frames.append(direct_expanded_reference_df)

    for estimator_name, cols in estimators.items():
        estimator_uncertainty_df = _build_estimator_uncertainty_df(
            pointwise_df,
            confidence_col=str(cols["confidence_col"]),
            uncertainty_col=str(cols["uncertainty_col"]),
        )
        compare_frames.extend(
            _evaluate_ranking_rows(
                ranking_df,
                estimator_uncertainty_df,
                estimator_name=estimator_name,
                ranking_family_params=ranking_family_params,
                domain=domain,
                model=model,
                topk=args.topk,
            )
        )
        pairwise_frames, diagnostics_row = _evaluate_pairwise_rows(
            pairwise_df,
            ranking_df,
            estimator_uncertainty_df,
            estimator_name=estimator_name,
            aggregation_variant=args.pairwise_aggregation_variant,
            prior_weight=args.pairwise_prior_weight,
            loss_weight=args.pairwise_loss_weight,
            score_scale=args.pairwise_score_scale,
            domain=domain,
            model=model,
            topk=args.topk,
        )
        compare_frames.extend(pairwise_frames)
        diagnostics_row["pairwise_exp_name"] = args.pairwise_exp_name
        diagnostics_row["ranking_exp_name"] = args.ranking_exp_name
        diagnostics_row["pointwise_exp_name"] = args.pointwise_exp_name
        coverage_rows.append(diagnostics_row)

    compare_df = pd.concat(compare_frames, ignore_index=True, sort=False)
    summary_dir = Path(args.output_root) / "summary"
    save_table(compare_df, summary_dir / "week6_day3_estimator_compare.csv")

    coverage_df = pd.DataFrame(coverage_rows)
    save_table(coverage_df, summary_dir / "week6_day3_pairwise_coverage_compare.csv")

    print(f"[week6_day3] Saved estimator compare to: {summary_dir / 'week6_day3_estimator_compare.csv'}")
    print(
        f"[week6_day3] Saved pairwise coverage diagnostics to: "
        f"{summary_dir / 'week6_day3_pairwise_coverage_compare.csv'}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from main_rank_rerank import (
    UNCERTAINTY_SOURCE_DEFAULTS,
    build_result_row,
    infer_domain_name,
    infer_model_name,
    load_jsonl,
    resolve_uncertainty_path,
    save_jsonl,
    save_table,
)
from src.eval.ranking_task_metrics import (
    build_ranking_eval_frame,
    compute_ranking_exposure_distribution,
)
from src.methods.uncertainty_pairwise_aggregator import (
    SUPPORTED_PAIRWISE_AGG_VARIANTS,
    aggregate_pairwise_preferences,
    build_pairwise_preference_rows,
    build_pairwise_ranked_predictions,
    rank_pairwise_candidates,
    summarize_pairwise_rank_effect,
)
from src.methods.uncertainty_ranker import summarize_rerank_effect
from src.utils.paths import ensure_exp_dirs
from src.utils.reproducibility import set_global_seed


def _infer_uncertainty_exp_name(ranking_exp_name: str, pairwise_exp_name: str) -> str:
    if ranking_exp_name.endswith("_rank"):
        return ranking_exp_name[: -len("_rank")]
    if pairwise_exp_name.endswith("_pairwise"):
        return pairwise_exp_name[: -len("_pairwise")]
    return ranking_exp_name


def _maybe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _maybe_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _filter_predictions_to_event_ids(prediction_df: pd.DataFrame, event_ids: set[str]) -> pd.DataFrame:
    if prediction_df.empty:
        return prediction_df.copy()
    return prediction_df[prediction_df["source_event_id"].astype(str).isin(event_ids)].copy()


def _resolve_compare_rerank_specs(
    *,
    output_root: str | Path,
    ranking_exp_name: str,
    summary_filename: str,
) -> list[dict[str, str]]:
    summary_path = Path(output_root) / "summary" / summary_filename
    if not summary_path.exists():
        return []

    summary_df = pd.read_csv(summary_path)
    if summary_df.empty or "base_exp_name" not in summary_df.columns:
        return []

    desired_variants = [
        "nonlinear_structured_risk_rerank",
        "local_margin_swap_rerank",
        "structured_risk_plus_local_margin_swap_rerank",
    ]
    filtered_df = summary_df[
        (summary_df["base_exp_name"].astype(str) == str(ranking_exp_name))
        & (summary_df["rerank_variant"].astype(str).isin(desired_variants))
    ].copy()
    if filtered_df.empty:
        return []

    specs: list[dict[str, str]] = []
    for variant in desired_variants:
        variant_df = filtered_df[filtered_df["rerank_variant"].astype(str) == variant].copy()
        if variant_df.empty:
            continue
        variant_df = variant_df.sort_values(
            by=["NDCG@10", "MRR", "changed_ranking_fraction", "rerank_exp_name"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        best_row = variant_df.iloc[0].to_dict()
        specs.append(
            {
                "rerank_variant": str(best_row.get("rerank_variant", variant)),
                "rerank_exp_name": str(best_row.get("rerank_exp_name", "")),
                "method": str(best_row.get("method", f"uncertainty_aware_rank_rerank_{variant}")),
                "family": str(best_row.get("family", variant)),
                "search_exp_name": str(best_row.get("search_exp_name", "")),
                "search_note": str(best_row.get("search_note", "")),
            }
        )
    return specs


def _build_overlap_metrics_row(
    *,
    method_name: str,
    prediction_df: pd.DataFrame,
    ranking_subset_df: pd.DataFrame,
    topk: int,
    total_ranking_events: int,
    extra_metrics: dict[str, float] | None = None,
    lambda_penalty: float | None = None,
    uncertainty_source: str | None = None,
    rerank_variant: str | None = None,
    gate_topk: int | None = None,
    tau: float | None = None,
    gamma: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    delta: float | None = None,
    coverage_fallback_scale: float | None = None,
    eta: float | None = None,
    m_rel: float | None = None,
    m_unc: float | None = None,
    swap_a: float | None = None,
    swap_b: float | None = None,
) -> pd.DataFrame:
    metrics = {
        "compare_subset_event_count": float(len(prediction_df)),
        "total_ranking_event_count": float(total_ranking_events),
        "pairwise_supported_event_fraction": (
            float(len(prediction_df) / total_ranking_events) if total_ranking_events else float("nan")
        ),
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    return build_result_row(
        method_name=method_name,
        prediction_df=prediction_df,
        topk=topk,
        lambda_penalty=lambda_penalty,
        uncertainty_source=uncertainty_source,
        rerank_variant=rerank_variant,
        gate_topk=gate_topk,
        tau=tau,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        delta=delta,
        coverage_fallback_scale=coverage_fallback_scale,
        eta=eta,
        m_rel=m_rel,
        m_unc=m_unc,
        swap_a=swap_a,
        swap_b=swap_b,
        extra_metrics=metrics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise_exp_name", type=str, default="beauty_qwen_pairwise")
    parser.add_argument("--ranking_exp_name", type=str, default="beauty_qwen_rank")
    parser.add_argument("--new_exp_name", type=str, default=None)
    parser.add_argument("--uncertainty_exp_name", type=str, default=None)
    parser.add_argument("--pairwise_input_path", type=str, default=None)
    parser.add_argument("--ranking_input_path", type=str, default=None)
    parser.add_argument("--uncertainty_input_path", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument(
        "--uncertainty_source",
        type=str,
        default="pointwise_calibrated",
        choices=list(UNCERTAINTY_SOURCE_DEFAULTS.keys()),
    )
    parser.add_argument("--uncertainty_col", type=str, default=None)
    parser.add_argument("--uncertainty_confidence_col", type=str, default=None)
    parser.add_argument(
        "--aggregation_variant",
        type=str,
        default="weighted_win_count",
        choices=sorted(SUPPORTED_PAIRWISE_AGG_VARIANTS),
    )
    parser.add_argument("--prior_weight", type=float, default=0.2)
    parser.add_argument("--loss_weight", type=float, default=1.0)
    parser.add_argument("--score_scale", type=float, default=1.0)
    parser.add_argument("--compare_summary_file", type=str, default="week6_rank_rerank_variant_compare.csv")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    new_exp_name = args.new_exp_name or f"{args.pairwise_exp_name}_to_rank"
    uncertainty_exp_name = args.uncertainty_exp_name or _infer_uncertainty_exp_name(
        args.ranking_exp_name,
        args.pairwise_exp_name,
    )

    pairwise_paths = ensure_exp_dirs(args.pairwise_exp_name, args.output_root)
    ranking_paths = ensure_exp_dirs(args.ranking_exp_name, args.output_root)
    pairwise_rank_paths = ensure_exp_dirs(new_exp_name, args.output_root)

    pairwise_input_path = (
        Path(args.pairwise_input_path)
        if args.pairwise_input_path
        else pairwise_paths.predictions_dir / "pairwise_predictions.jsonl"
    )
    ranking_input_path = (
        Path(args.ranking_input_path)
        if args.ranking_input_path
        else ranking_paths.predictions_dir / "rank_predictions.jsonl"
    )
    uncertainty_input_path = resolve_uncertainty_path(
        output_root=args.output_root,
        uncertainty_exp_name=uncertainty_exp_name,
        uncertainty_source=args.uncertainty_source,
        explicit_path=args.uncertainty_input_path,
    )

    if not pairwise_input_path.exists():
        raise FileNotFoundError(f"Pairwise prediction file not found: {pairwise_input_path}")
    if not ranking_input_path.exists():
        raise FileNotFoundError(f"Ranking prediction file not found: {ranking_input_path}")
    if not uncertainty_input_path.exists():
        raise FileNotFoundError(f"Uncertainty file not found: {uncertainty_input_path}")

    source_defaults = UNCERTAINTY_SOURCE_DEFAULTS[args.uncertainty_source]
    uncertainty_col = args.uncertainty_col or source_defaults["uncertainty_col"]
    uncertainty_confidence_col = args.uncertainty_confidence_col or source_defaults["confidence_col"]

    print(f"[{args.pairwise_exp_name}] Loading pairwise predictions from: {pairwise_input_path}")
    pairwise_df = load_jsonl(pairwise_input_path)
    print(f"[{args.pairwise_exp_name}] Loaded {len(pairwise_df)} pairwise rows.")

    print(f"[{args.ranking_exp_name}] Loading ranking predictions from: {ranking_input_path}")
    ranking_df = load_jsonl(ranking_input_path)
    print(f"[{args.ranking_exp_name}] Loaded {len(ranking_df)} ranking events.")

    print(f"[{uncertainty_exp_name}] Loading uncertainty source from: {uncertainty_input_path}")
    uncertainty_df = load_jsonl(uncertainty_input_path)
    print(f"[{uncertainty_exp_name}] Loaded {len(uncertainty_df)} uncertainty rows.")

    pairwise_rows_df = build_pairwise_preference_rows(
        pairwise_predictions_df=pairwise_df,
        ranking_predictions_df=ranking_df,
        uncertainty_df=uncertainty_df,
        uncertainty_col=uncertainty_col,
        uncertainty_confidence_col=uncertainty_confidence_col,
        uncertainty_source=args.uncertainty_source,
    )
    if pairwise_rows_df.empty:
        raise RuntimeError(
            "No pairwise rows could be aligned with ranking events. "
            "Check whether pairwise and ranking experiments share source_event_id."
        )

    aggregated_item_df = aggregate_pairwise_preferences(
        pairwise_rows_df,
        ranking_df,
        aggregation_variant=args.aggregation_variant,
        prior_weight=args.prior_weight,
        loss_weight=args.loss_weight,
        score_scale=args.score_scale,
    )
    ranked_item_df = rank_pairwise_candidates(aggregated_item_df)
    pairwise_ranked_predictions = build_pairwise_ranked_predictions(
        ranked_item_df,
        ranking_df,
        topk=args.k,
        total_ranking_events=len(ranking_df),
    )

    support_event_ids = set(pairwise_ranked_predictions["source_event_id"].astype(str).tolist())
    ranking_subset_df = _filter_predictions_to_event_ids(ranking_df, support_event_ids)
    if ranking_subset_df.empty:
        raise RuntimeError("Pairwise-to-rank overlap subset is empty after aggregation.")

    save_table(pairwise_rows_df, pairwise_rank_paths.tables_dir / "pairwise_preference_rows.csv")
    save_table(ranked_item_df, pairwise_rank_paths.tables_dir / "pairwise_rank_item_scores.csv")
    save_jsonl(pairwise_ranked_predictions, pairwise_rank_paths.reranked_dir / "pairwise_ranked.jsonl")

    baseline_row = _build_overlap_metrics_row(
        method_name="direct_candidate_ranking",
        prediction_df=ranking_subset_df,
        ranking_subset_df=ranking_subset_df,
        topk=args.k,
        total_ranking_events=len(ranking_df),
        extra_metrics={
            "avg_uncertainty_coverage_rate": float("nan"),
            "uncertainty_coverage": float("nan"),
            "changed_ranking_fraction": 0.0,
            "avg_position_shift": 0.0,
            "pairwise_pair_coverage_rate": float("nan"),
            "pairwise_supported_candidate_fraction": float("nan"),
            "pairwise_avg_reliability_weight": float("nan"),
        },
    )
    baseline_row["family"] = "direct_candidate_ranking"
    baseline_row["pairwise_aggregation_variant"] = None
    baseline_row["prior_weight"] = None
    baseline_row["loss_weight"] = None
    baseline_row["score_scale"] = None
    baseline_row["base_exp_name"] = args.ranking_exp_name
    baseline_row["rerank_exp_name"] = args.ranking_exp_name
    baseline_row["pairwise_exp_name"] = args.pairwise_exp_name
    baseline_row["uncertainty_exp_name"] = uncertainty_exp_name
    baseline_row["evaluation_scope"] = "pairwise_event_overlap_subset"

    pairwise_effect_metrics = summarize_pairwise_rank_effect(
        ranking_subset_df,
        pairwise_ranked_predictions,
        total_ranking_events=len(ranking_df),
    )
    pairwise_row = _build_overlap_metrics_row(
        method_name=f"pairwise_to_rank_{args.aggregation_variant}",
        prediction_df=pairwise_ranked_predictions,
        ranking_subset_df=ranking_subset_df,
        topk=args.k,
        total_ranking_events=len(ranking_df),
        uncertainty_source=args.uncertainty_source,
        extra_metrics=pairwise_effect_metrics,
    )
    pairwise_row["family"] = "pairwise_to_rank"
    pairwise_row["pairwise_aggregation_variant"] = args.aggregation_variant
    pairwise_row["prior_weight"] = float(args.prior_weight)
    pairwise_row["loss_weight"] = float(args.loss_weight)
    pairwise_row["score_scale"] = float(args.score_scale)
    pairwise_row["base_exp_name"] = args.ranking_exp_name
    pairwise_row["rerank_exp_name"] = new_exp_name
    pairwise_row["pairwise_exp_name"] = args.pairwise_exp_name
    pairwise_row["uncertainty_exp_name"] = uncertainty_exp_name
    pairwise_row["evaluation_scope"] = "pairwise_event_overlap_subset"

    results_df = pd.concat([baseline_row, pairwise_row], ignore_index=True)
    save_table(results_df, pairwise_rank_paths.tables_dir / "rerank_results.csv")

    baseline_exposure_df = compute_ranking_exposure_distribution(build_ranking_eval_frame(ranking_subset_df), k=args.k)
    baseline_exposure_df["method"] = "direct_candidate_ranking"
    pairwise_exposure_df = compute_ranking_exposure_distribution(
        build_ranking_eval_frame(pairwise_ranked_predictions),
        k=args.k,
    )
    pairwise_exposure_df["method"] = f"pairwise_to_rank_{args.aggregation_variant}"
    exposure_df = pd.concat([baseline_exposure_df, pairwise_exposure_df], ignore_index=True)
    save_table(exposure_df, pairwise_rank_paths.tables_dir / "topk_exposure_distribution.csv")

    compare_frames = [baseline_row.copy()]
    compare_specs = _resolve_compare_rerank_specs(
        output_root=args.output_root,
        ranking_exp_name=args.ranking_exp_name,
        summary_filename=args.compare_summary_file,
    )
    for spec in compare_specs:
        rerank_exp_name = spec["rerank_exp_name"]
        compare_path = Path(args.output_root) / rerank_exp_name / "reranked" / "rank_reranked.jsonl"
        if not compare_path.exists():
            continue

        compare_prediction_df = load_jsonl(compare_path)
        compare_prediction_subset_df = _filter_predictions_to_event_ids(compare_prediction_df, support_event_ids)
        if compare_prediction_subset_df.empty:
            continue

        first_record = compare_prediction_subset_df.iloc[0].to_dict()
        compare_effect_metrics = summarize_rerank_effect(ranking_subset_df, compare_prediction_subset_df)
        compare_row = _build_overlap_metrics_row(
            method_name=str(spec["method"]),
            prediction_df=compare_prediction_subset_df,
            ranking_subset_df=ranking_subset_df,
            topk=args.k,
            total_ranking_events=len(ranking_df),
            lambda_penalty=_maybe_float(first_record.get("lambda_penalty")),
            uncertainty_source=str(first_record.get("uncertainty_source", "")) or None,
            rerank_variant=str(first_record.get("rerank_variant", "")) or None,
            gate_topk=_maybe_int(first_record.get("gate_topk")),
            tau=_maybe_float(first_record.get("tau")),
            gamma=_maybe_float(first_record.get("gamma")),
            alpha=_maybe_float(first_record.get("alpha")),
            beta=_maybe_float(first_record.get("beta")),
            delta=_maybe_float(first_record.get("delta")),
            coverage_fallback_scale=_maybe_float(first_record.get("coverage_fallback_scale")),
            eta=_maybe_float(first_record.get("eta")),
            m_rel=_maybe_float(first_record.get("m_rel")),
            m_unc=_maybe_float(first_record.get("m_unc")),
            swap_a=_maybe_float(first_record.get("swap_a")),
            swap_b=_maybe_float(first_record.get("swap_b")),
            extra_metrics=compare_effect_metrics,
        )
        compare_row["family"] = spec["family"]
        compare_row["pairwise_aggregation_variant"] = None
        compare_row["prior_weight"] = None
        compare_row["loss_weight"] = None
        compare_row["score_scale"] = None
        compare_row["base_exp_name"] = args.ranking_exp_name
        compare_row["rerank_exp_name"] = rerank_exp_name
        compare_row["pairwise_exp_name"] = args.pairwise_exp_name
        compare_row["uncertainty_exp_name"] = uncertainty_exp_name
        compare_row["evaluation_scope"] = "pairwise_event_overlap_subset"
        compare_row["search_exp_name"] = spec.get("search_exp_name", "")
        compare_row["search_note"] = spec.get("search_note", "")
        compare_frames.append(compare_row)

    compare_frames.append(pairwise_row.copy())
    compare_df = pd.concat(compare_frames, ignore_index=True)
    compare_df.insert(0, "model", infer_model_name(args.ranking_exp_name))
    compare_df.insert(0, "domain", infer_domain_name(args.ranking_exp_name))
    compare_df.insert(2, "task", "candidate_ranking")

    summary_dir = Path(args.output_root) / "summary"
    save_table(compare_df, summary_dir / f"{new_exp_name}_compare.csv")
    if new_exp_name == f"{args.pairwise_exp_name}_to_rank":
        save_table(compare_df, summary_dir / "week6_day2_pairwise_to_rank_compare.csv")

    print(f"[{new_exp_name}] Pairwise-to-rank done.")
    print(
        f"[{new_exp_name}] Overlap events: {len(support_event_ids)}/{len(ranking_df)} "
        f"({len(support_event_ids) / len(ranking_df):.2%})"
    )
    print(f"[{new_exp_name}] Predictions saved to: {pairwise_rank_paths.reranked_dir / 'pairwise_ranked.jsonl'}")
    print(f"[{new_exp_name}] Tables saved to: {pairwise_rank_paths.tables_dir}")


if __name__ == "__main__":
    main()

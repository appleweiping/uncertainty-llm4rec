from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from main_rank_rerank import save_table
from src.methods.baseline_ranker_multitask import (
    build_pairwise_plain_baseline_rows,
    build_pointwise_baseline_compare_rows,
    select_best_existing_row,
    _decorate_compare_row,
    _ensure_compare_columns,
)
from src.utils.reproducibility import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointwise_exp_name", type=str, default="beauty_qwen")
    parser.add_argument("--ranking_exp_name", type=str, default="beauty_qwen_rank")
    parser.add_argument("--pairwise_exp_name", type=str, default="beauty_qwen_pairwise")
    parser.add_argument("--pairwise_baseline_exp_name", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument(
        "--day3_compare_path",
        type=str,
        default="outputs/summary/week6_day3_estimator_compare.csv",
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--pairwise_prior_weight", type=float, default=0.2)
    parser.add_argument("--pairwise_loss_weight", type=float, default=1.0)
    parser.add_argument("--pairwise_score_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _finalize_row(
    row_df: pd.DataFrame,
    *,
    is_same_task_baseline: bool,
    is_current_best_family: bool,
    notes: str | None = None,
) -> pd.DataFrame:
    out = _ensure_compare_columns(row_df)
    out = _decorate_compare_row(
        out,
        is_same_task_baseline=is_same_task_baseline,
        is_current_best_family=is_current_best_family,
        notes=notes,
    )
    return out


def _concat_with_schema(frames: list[pd.DataFrame], schema_columns: list[str]) -> pd.DataFrame:
    normalized_frames: list[pd.DataFrame] = []
    for frame in frames:
        out = frame.copy()
        for column in schema_columns:
            if column not in out.columns:
                out[column] = pd.NA
        normalized_frames.append(out[schema_columns])
    return pd.concat(normalized_frames, ignore_index=True, sort=False)


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    pairwise_baseline_exp_name = (
        args.pairwise_baseline_exp_name or f"{args.pairwise_exp_name}_plain_to_rank"
    )
    day3_compare_path = Path(args.day3_compare_path)
    if not day3_compare_path.exists():
        raise FileNotFoundError(
            f"Day3 compare file not found: {day3_compare_path}. "
            "Run main_uncertainty_compare_multitask.py before day4 baseline compare."
        )

    day3_compare_df = pd.read_csv(day3_compare_path)
    base_schema = list(day3_compare_df.columns)
    for extra_column in ["is_same_task_baseline", "is_current_best_family"]:
        if extra_column not in base_schema:
            base_schema.append(extra_column)

    pointwise_df = build_pointwise_baseline_compare_rows(
        output_root=args.output_root,
        pointwise_exp_name=args.pointwise_exp_name,
    )
    plain_pairwise_df = build_pairwise_plain_baseline_rows(
        output_root=args.output_root,
        pointwise_exp_name=args.pointwise_exp_name,
        ranking_exp_name=args.ranking_exp_name,
        pairwise_exp_name=args.pairwise_exp_name,
        pairwise_baseline_exp_name=pairwise_baseline_exp_name,
        topk=args.topk,
        prior_weight=args.pairwise_prior_weight,
        loss_weight=args.pairwise_loss_weight,
        score_scale=args.pairwise_score_scale,
    )

    selected_frames: list[pd.DataFrame] = []

    direct_ranking_df = select_best_existing_row(
        day3_compare_df,
        task="candidate_ranking",
        evaluation_scope="full_ranking_set",
        method_variant="direct_candidate_ranking",
    )
    selected_frames.append(
        _finalize_row(
            direct_ranking_df,
            is_same_task_baseline=True,
            is_current_best_family=False,
            notes="same-task direct ranking baseline without uncertainty",
        )
    )

    structured_df = select_best_existing_row(
        day3_compare_df,
        task="candidate_ranking",
        evaluation_scope="full_ranking_set",
        method_family="structured_risk_family",
    )
    selected_frames.append(
        _finalize_row(
            structured_df,
            is_same_task_baseline=False,
            is_current_best_family=True,
            notes="current best ranking family retained as the main decision line",
        )
    )

    local_df = select_best_existing_row(
        day3_compare_df,
        task="candidate_ranking",
        evaluation_scope="full_ranking_set",
        method_family="local_margin_swap_family",
    )
    selected_frames.append(
        _finalize_row(
            local_df,
            is_same_task_baseline=False,
            is_current_best_family=False,
            notes="retained local swap family for same-task baseline comparison",
        )
    )

    combo_df = select_best_existing_row(
        day3_compare_df,
        task="candidate_ranking",
        evaluation_scope="full_ranking_set",
        method_family="structured_risk_plus_local_swap_family",
    )
    selected_frames.append(
        _finalize_row(
            combo_df,
            is_same_task_baseline=False,
            is_current_best_family=False,
            notes="retained complex family kept visible in the same baseline matrix",
        )
    )

    pairwise_direct_overlap_df = select_best_existing_row(
        day3_compare_df,
        task="pairwise_to_rank",
        evaluation_scope="pairwise_event_overlap_subset",
        method_variant="direct_overlap_reference",
    )
    selected_frames.append(
        _finalize_row(
            pairwise_direct_overlap_df,
            is_same_task_baseline=True,
            is_current_best_family=False,
            notes="direct ranking reference on pairwise-supported overlap subset",
        )
    )

    pairwise_weighted_overlap_df = select_best_existing_row(
        day3_compare_df,
        task="pairwise_to_rank",
        evaluation_scope="pairwise_event_overlap_subset",
        method_family="pairwise_to_rank",
    )
    selected_frames.append(
        _finalize_row(
            pairwise_weighted_overlap_df,
            is_same_task_baseline=False,
            is_current_best_family=False,
            notes="uncertainty-aware pairwise aggregation under overlap subset",
        )
    )

    pairwise_direct_expanded_df = select_best_existing_row(
        day3_compare_df,
        task="pairwise_to_rank",
        evaluation_scope="expanded_with_direct_fallback",
        method_variant="direct_expanded_reference",
    )
    selected_frames.append(
        _finalize_row(
            pairwise_direct_expanded_df,
            is_same_task_baseline=True,
            is_current_best_family=False,
            notes="direct ranking reference on expanded full set before pairwise fallback compare",
        )
    )

    pairwise_weighted_expanded_df = select_best_existing_row(
        day3_compare_df,
        task="pairwise_to_rank",
        evaluation_scope="expanded_with_direct_fallback",
        method_family="pairwise_to_rank",
    )
    selected_frames.append(
        _finalize_row(
            pairwise_weighted_expanded_df,
            is_same_task_baseline=False,
            is_current_best_family=False,
            notes="uncertainty-aware pairwise aggregation with direct-ranking fallback",
        )
    )

    compare_frames = [pointwise_df] + selected_frames + [plain_pairwise_df]
    compare_df = _concat_with_schema(compare_frames, base_schema)

    sort_task_order = {
        "pointwise_yesno": 0,
        "candidate_ranking": 1,
        "pairwise_to_rank": 2,
    }
    sort_scope_order = {
        "full_pointwise_set": 0,
        "full_ranking_set": 1,
        "pairwise_event_overlap_subset": 2,
        "expanded_with_direct_fallback": 3,
    }
    compare_df["_task_order"] = compare_df["task"].map(sort_task_order).fillna(99)
    compare_df["_scope_order"] = compare_df["evaluation_scope"].map(sort_scope_order).fillna(99)
    compare_df["_baseline_order"] = compare_df["is_same_task_baseline"].astype(bool).map({True: 0, False: 1})
    compare_df = compare_df.sort_values(
        by=["_task_order", "_scope_order", "_baseline_order", "method_family", "method_variant"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    ).drop(columns=["_task_order", "_scope_order", "_baseline_order"])

    output_path = Path(args.output_root) / "summary" / "week6_day4_decision_baseline_compare.csv"
    save_table(compare_df, output_path)

    print(f"[week6-day4] Baseline compare saved to: {output_path}")
    print(f"[week6-day4] Rows: {len(compare_df)}")


if __name__ == "__main__":
    main()

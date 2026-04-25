from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from main_rerank import load_jsonl, save_table
from main_rerank_grid import parse_lambda_grid, usable_rerank_rows, validate_input_columns
from src.analysis.rerank_diagnostics import compute_rank_change_diagnostics
from src.methods.baseline_ranker import add_baseline_score, rank_by_score
from src.methods.uncertainty_reranker import rerank_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_exp_name", type=str, required=True)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="output-repaired")
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--lambda_grid", type=str, default="0,0.1,0.2,0.5,1.0")
    parser.add_argument("--score_column", type=str, default="minimal_repaired_confidence")
    parser.add_argument("--uncertainty_column", type=str, default="minimal_evidence_uncertainty")
    parser.add_argument("--setting", type=str, default="monotonic_repaired_confidence_penalty")
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = (
        Path(args.input_path)
        if args.input_path
        else Path(args.output_root) / args.base_exp_name / "calibrated" / "evidence_posterior_test.jsonl"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Rerank diagnostic input not found: {input_path}")

    raw_df = load_jsonl(input_path)
    validate_input_columns(
        raw_df,
        [
            "user_id",
            "candidate_item_id",
            "label",
            "target_popularity_group",
            args.score_column,
            args.uncertainty_column,
        ],
    )
    df = usable_rerank_rows(raw_df, score_column=args.score_column, uncertainty_column=args.uncertainty_column)
    if df.empty:
        raise ValueError("No usable rows remain for rerank diagnostics.")

    baseline_df = add_baseline_score(df, score_col=args.score_column, output_col="baseline_score")
    baseline_ranked = rank_by_score(
        baseline_df,
        user_col="user_id",
        score_col="baseline_score",
        rank_col="rank",
    )

    rows = []
    for lambda_penalty in parse_lambda_grid(args.lambda_grid):
        rerank_ranked = rerank_candidates(
            df=df,
            user_col="user_id",
            confidence_col=args.score_column,
            uncertainty_col=args.uncertainty_column,
            lambda_penalty=lambda_penalty,
            score_col="final_score",
            rank_col="rank",
        )
        rows.append(
            compute_rank_change_diagnostics(
                baseline_ranked,
                rerank_ranked,
                base_score_col=args.score_column,
                uncertainty_col=args.uncertainty_column,
                lambda_penalty=lambda_penalty,
                setting=args.setting,
                normalization="none",
                k=args.k,
            )
        )

    summary_df = pd.DataFrame(rows)
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else Path(args.output_root) / "summary" / f"{args.base_exp_name}_rerank_diagnostics.csv"
    )
    save_table(summary_df, summary_path)
    print(f"Saved rerank diagnostics to: {summary_path}")
    if bool(summary_df["rerank_is_noop"].all()):
        print(
            "All lambda settings are no-op: the current score/uncertainty pair does not change ranking order."
        )


if __name__ == "__main__":
    main()

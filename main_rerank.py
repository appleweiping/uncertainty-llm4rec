# main_rerank.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval.bias_metrics import (
    compute_bias_metrics,
    compute_topk_exposure_distribution,
)
from src.eval.ranking_metrics import compute_ranking_metrics
from src.methods.baseline_ranker import add_baseline_score, rank_by_score
from src.methods.uncertainty_reranker import (
    add_uncertainty_aware_score,
    rank_by_rerank_score,
)


def load_jsonl(path: str | Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def save_jsonl(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_result_row(
    method_name: str,
    ranked_df: pd.DataFrame,
    k: int
) -> pd.DataFrame:
    ranking = compute_ranking_metrics(ranked_df, k=k)
    bias = compute_bias_metrics(ranked_df, k=k)

    row = {"method": method_name}
    row.update(ranking)
    row.update(bias)
    return pd.DataFrame([row])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="outputs/calibrated/test_calibrated.jsonl",
        help="Path to calibrated prediction jsonl file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Base output directory."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-K for ranking evaluation."
    )
    parser.add_argument(
        "--lambda_penalty",
        type=float,
        default=0.5,
        help="Penalty coefficient for uncertainty-aware reranking."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    reranked_dir = output_dir / "reranked"
    tables_dir = output_dir / "tables"

    reranked_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading calibrated predictions from: {args.input_path}")
    df = load_jsonl(args.input_path)
    print(f"Loaded {len(df)} samples.")

    required_cols = [
        "user_id",
        "candidate_item_id",
        "label",
        "target_popularity_group",
        "calibrated_confidence",
        "uncertainty",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column `{col}` not found in input file.")

    # Baseline ranking
    baseline_df = add_baseline_score(
        df,
        score_col="calibrated_confidence",
        output_col="baseline_score"
    )
    baseline_ranked = rank_by_score(
        baseline_df,
        user_col="user_id",
        score_col="baseline_score",
        rank_col="rank"
    )
    save_jsonl(baseline_ranked, reranked_dir / "baseline_ranked.jsonl")
    print("Saved baseline_ranked.jsonl")

    # Uncertainty-aware reranking
    rerank_df = add_uncertainty_aware_score(
        df,
        confidence_col="calibrated_confidence",
        uncertainty_col="uncertainty",
        lambda_penalty=args.lambda_penalty,
        output_col="rerank_score"
    )
    rerank_ranked = rank_by_rerank_score(
        rerank_df,
        user_col="user_id",
        score_col="rerank_score",
        rank_col="rank"
    )
    save_jsonl(rerank_ranked, reranked_dir / "uncertainty_reranked.jsonl")
    print("Saved uncertainty_reranked.jsonl")

    # Results table
    baseline_row = build_result_row("baseline", baseline_ranked, k=args.k)
    rerank_row = build_result_row("uncertainty_aware_rerank", rerank_ranked, k=args.k)

    results_df = pd.concat([baseline_row, rerank_row], ignore_index=True)
    save_table(results_df, tables_dir / "rerank_results.csv")
    print("Saved rerank_results.csv")

    # Exposure distribution tables
    baseline_dist = compute_topk_exposure_distribution(baseline_ranked, k=args.k)
    baseline_dist["method"] = "baseline"
    rerank_dist = compute_topk_exposure_distribution(rerank_ranked, k=args.k)
    rerank_dist["method"] = "uncertainty_aware_rerank"

    exposure_dist_df = pd.concat([baseline_dist, rerank_dist], ignore_index=True)
    save_table(exposure_dist_df, tables_dir / "topk_exposure_distribution.csv")
    print("Saved topk_exposure_distribution.csv")

    print("\nDone.")
    print("Main outputs:")
    print(f"- {reranked_dir / 'baseline_ranked.jsonl'}")
    print(f"- {reranked_dir / 'uncertainty_reranked.jsonl'}")
    print(f"- {tables_dir / 'rerank_results.csv'}")
    print(f"- {tables_dir / 'topk_exposure_distribution.csv'}")


if __name__ == "__main__":
    main()
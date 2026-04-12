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
from src.methods.uncertainty_reranker import rerank_candidates
from src.utils.paths import ensure_exp_dirs
from src.utils.reproducibility import set_global_seed


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
    k: int,
    lambda_penalty: float | None = None,
) -> pd.DataFrame:
    ranking = compute_ranking_metrics(ranked_df, k=k)
    bias = compute_bias_metrics(ranked_df, k=k)

    row = {"method": method_name}
    if lambda_penalty is not None:
        row["lambda_penalty"] = float(lambda_penalty)
    row.update(ranking)
    row.update(bias)
    return pd.DataFrame([row])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="clean",
        help="Experiment name."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Optional explicit path to calibrated prediction jsonl. Defaults to outputs/{exp_name}/calibrated/test_calibrated.jsonl"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs",
        help="Root directory for all experiment outputs."
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional global random seed."
    )
    args = parser.parse_args()
    set_global_seed(args.seed)

    paths = ensure_exp_dirs(args.exp_name, args.output_root)
    input_path = (
        Path(args.input_path)
        if args.input_path is not None
        else paths.calibrated_dir / "test_calibrated.jsonl"
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Calibrated file not found: {input_path}")

    print(f"[{args.exp_name}] Loading calibrated predictions from: {input_path}")
    df = load_jsonl(input_path)
    print(f"[{args.exp_name}] Loaded {len(df)} samples.")
    if args.seed is not None:
        print(f"[{args.exp_name}] Seed: {args.seed}")

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
    save_jsonl(baseline_ranked, paths.reranked_dir / "baseline_ranked.jsonl")

    rerank_ranked = rerank_candidates(
        df=df,
        user_col="user_id",
        confidence_col="calibrated_confidence",
        uncertainty_col="uncertainty",
        lambda_penalty=args.lambda_penalty,
        score_col="final_score",
        rank_col="rank",
    )
    save_jsonl(rerank_ranked, paths.reranked_dir / "uncertainty_reranked.jsonl")

    baseline_row = build_result_row("baseline", baseline_ranked, k=args.k)
    rerank_row = build_result_row(
        "uncertainty_aware_rerank",
        rerank_ranked,
        k=args.k,
        lambda_penalty=args.lambda_penalty,
    )

    results_df = pd.concat([baseline_row, rerank_row], ignore_index=True)
    save_table(results_df, paths.tables_dir / "rerank_results.csv")
    save_table(results_df, paths.reranked_dir / "rerank_results.csv")

    baseline_dist = compute_topk_exposure_distribution(baseline_ranked, k=args.k)
    baseline_dist["method"] = "baseline"
    rerank_dist = compute_topk_exposure_distribution(rerank_ranked, k=args.k)
    rerank_dist["method"] = "uncertainty_aware_rerank"

    exposure_dist_df = pd.concat([baseline_dist, rerank_dist], ignore_index=True)
    save_table(exposure_dist_df, paths.tables_dir / "topk_exposure_distribution.csv")

    print(f"[{args.exp_name}] Reranking done.")
    print(f"Reranked files saved to: {paths.reranked_dir}")
    print(f"Tables saved to:         {paths.tables_dir}")


if __name__ == "__main__":
    main()

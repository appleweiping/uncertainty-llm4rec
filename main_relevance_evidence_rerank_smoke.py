from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from main_rerank import build_result_row, load_jsonl, save_jsonl, save_table
from main_rerank_grid import lambda_slug, parse_lambda_grid, usable_rerank_rows, validate_input_columns
from src.analysis.rerank_diagnostics import add_user_normalized_column, compute_rank_change_diagnostics
from src.eval.bias_metrics import compute_topk_exposure_distribution
from src.methods.baseline_ranker import add_baseline_score, rank_by_score
from src.methods.uncertainty_reranker import rank_by_rerank_score
from src.utils.paths import ensure_exp_dirs


def _prepare_frame(df: pd.DataFrame, normalization: str) -> pd.DataFrame:
    out = add_user_normalized_column(
        df,
        "relevance_probability",
        "normalized_base_score",
        method=normalization,
    )
    out = add_user_normalized_column(
        out,
        "evidence_risk",
        "normalized_uncertainty",
        method=normalization,
    )
    return out


def run_lambda(
    *,
    df: pd.DataFrame,
    exp_name: str,
    output_root: str,
    lambda_penalty: float,
    normalization: str,
    k: int,
) -> pd.DataFrame:
    paths = ensure_exp_dirs(exp_name, output_root)
    prepared = _prepare_frame(df, normalization=normalization)

    baseline_df = add_baseline_score(prepared, score_col="normalized_base_score", output_col="baseline_score")
    baseline_ranked = rank_by_score(
        baseline_df,
        user_col="user_id",
        score_col="baseline_score",
        rank_col="rank",
    )
    save_jsonl(baseline_ranked, paths.reranked_dir / "baseline_ranked.jsonl")

    rerank_scored = prepared.copy()
    rerank_scored["final_score"] = (
        rerank_scored["normalized_base_score"].astype(float)
        - float(lambda_penalty) * rerank_scored["normalized_uncertainty"].astype(float)
    )
    rerank_ranked = rank_by_rerank_score(
        rerank_scored,
        user_col="user_id",
        score_col="final_score",
        rank_col="rank",
    )
    save_jsonl(rerank_ranked, paths.reranked_dir / "relevance_evidence_reranked.jsonl")

    baseline_row = build_result_row("baseline_relevance_probability", baseline_ranked, k=k)
    rerank_row = build_result_row(
        "relevance_evidence_decoupled_rerank",
        rerank_ranked,
        k=k,
        lambda_penalty=lambda_penalty,
    )
    results = pd.concat([baseline_row, rerank_row], ignore_index=True)
    diagnostics = compute_rank_change_diagnostics(
        baseline_ranked,
        rerank_ranked,
        base_score_col="relevance_probability",
        uncertainty_col="evidence_risk",
        lambda_penalty=lambda_penalty,
        setting="relevance_B",
        normalization=normalization,
        k=k,
    )
    for key, value in diagnostics.items():
        results[key] = value
    results["exp_name"] = exp_name
    results["base_score_col"] = "relevance_probability"
    results["uncertainty_col"] = "evidence_risk"
    results["normalization"] = normalization
    save_table(results, paths.tables_dir / "rerank_results.csv")
    save_table(results, paths.reranked_dir / "rerank_results.csv")

    baseline_dist = compute_topk_exposure_distribution(baseline_ranked, k=k)
    baseline_dist["method"] = "baseline_relevance_probability"
    baseline_dist["lambda_penalty"] = float(lambda_penalty)
    rerank_dist = compute_topk_exposure_distribution(rerank_ranked, k=k)
    rerank_dist["method"] = "relevance_evidence_decoupled_rerank"
    rerank_dist["lambda_penalty"] = float(lambda_penalty)
    save_table(
        pd.concat([baseline_dist, rerank_dist], ignore_index=True),
        paths.tables_dir / "topk_exposure_distribution.csv",
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="beauty_deepseek_relevance_evidence_100")
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="output-repaired")
    parser.add_argument(
        "--summary_path",
        type=str,
        default="output-repaired/summary/beauty_day7_relevance_evidence_rerank_smoke.csv",
    )
    parser.add_argument("--lambda_grid", type=str, default="0,0.1,0.2")
    parser.add_argument("--normalization", type=str, default="minmax", choices=["minmax", "zscore", "none"])
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = (
        Path(args.input_path)
        if args.input_path
        else Path(args.output_root) / args.exp_name / "calibrated" / "relevance_evidence_posterior_test.jsonl"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Relevance evidence rerank input not found: {input_path}")

    raw_df = load_jsonl(input_path)
    required_cols = [
        "user_id",
        "candidate_item_id",
        "label",
        "target_popularity_group",
        "relevance_probability",
        "evidence_risk",
    ]
    validate_input_columns(raw_df, required_cols)
    df = usable_rerank_rows(raw_df, score_column="relevance_probability", uncertainty_column="evidence_risk")
    if df.empty:
        raise ValueError("No usable relevance evidence rows remain for rerank smoke test.")

    lambdas = parse_lambda_grid(args.lambda_grid)
    rows: list[pd.DataFrame] = []
    for lambda_penalty in lambdas:
        exp_name = f"{args.exp_name}_relevance_rerank_{args.normalization}_{lambda_slug(lambda_penalty)}"
        result = run_lambda(
            df=df,
            exp_name=exp_name,
            output_root=args.output_root,
            lambda_penalty=lambda_penalty,
            normalization=args.normalization,
            k=args.k,
        )
        result["base_exp_name"] = args.exp_name
        result["input_path"] = str(input_path)
        result["usable_rows"] = int(len(df))
        result["total_rows"] = int(len(raw_df))
        rows.append(result)

    summary = pd.concat(rows, ignore_index=True)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    save_table(summary, summary_path)
    print(f"[{args.exp_name}] Relevance evidence rerank smoke done.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

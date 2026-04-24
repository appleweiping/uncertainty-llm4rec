from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from main_rerank import build_result_row, load_jsonl, save_jsonl, save_table
from src.eval.bias_metrics import compute_topk_exposure_distribution
from src.methods.baseline_ranker import add_baseline_score, rank_by_score
from src.methods.uncertainty_reranker import rerank_candidates
from src.utils.paths import ensure_exp_dirs
from src.utils.reproducibility import set_global_seed


def parse_lambda_grid(value: str) -> list[float]:
    lambdas: list[float] = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        lambdas.append(float(text))
    if not lambdas:
        raise ValueError("At least one lambda value is required.")
    return lambdas


def lambda_slug(value: float) -> str:
    if value == 0:
        return "l0"
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return f"l{text}"


def validate_input_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [column for column in required_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from rerank input: {missing}")


def usable_rerank_rows(df: pd.DataFrame, score_column: str, uncertainty_column: str) -> pd.DataFrame:
    out = df.copy()
    if "parse_success" in out.columns:
        out = out[out["parse_success"].astype(bool)].copy()
    out[score_column] = pd.to_numeric(out[score_column], errors="coerce")
    out[uncertainty_column] = pd.to_numeric(out[uncertainty_column], errors="coerce")
    out = out.dropna(
        subset=[
            "user_id",
            "candidate_item_id",
            "label",
            "target_popularity_group",
            score_column,
            uncertainty_column,
        ]
    ).copy()
    return out.reset_index(drop=True)


def run_single_lambda(
    *,
    df: pd.DataFrame,
    exp_name: str,
    output_root: str,
    k: int,
    lambda_penalty: float,
    score_column: str,
    uncertainty_column: str,
    method_variant: str,
) -> pd.DataFrame:
    paths = ensure_exp_dirs(exp_name, output_root)

    baseline_df = add_baseline_score(df, score_col=score_column, output_col="baseline_score")
    baseline_ranked = rank_by_score(
        baseline_df,
        user_col="user_id",
        score_col="baseline_score",
        rank_col="rank",
    )
    save_jsonl(baseline_ranked, paths.reranked_dir / "baseline_ranked.jsonl")

    rerank_ranked = rerank_candidates(
        df=df,
        user_col="user_id",
        confidence_col=score_column,
        uncertainty_col=uncertainty_column,
        lambda_penalty=lambda_penalty,
        score_col="final_score",
        rank_col="rank",
    )
    save_jsonl(rerank_ranked, paths.reranked_dir / "uncertainty_reranked.jsonl")

    baseline_row = build_result_row("baseline", baseline_ranked, k=k)
    rerank_row = build_result_row(
        "uncertainty_aware_rerank",
        rerank_ranked,
        k=k,
        lambda_penalty=lambda_penalty,
    )
    results_df = pd.concat([baseline_row, rerank_row], ignore_index=True)
    results_df["exp_name"] = exp_name
    results_df["method_variant"] = method_variant
    results_df["score_column"] = score_column
    results_df["uncertainty_column"] = uncertainty_column
    save_table(results_df, paths.tables_dir / "rerank_results.csv")
    save_table(results_df, paths.reranked_dir / "rerank_results.csv")

    baseline_dist = compute_topk_exposure_distribution(baseline_ranked, k=k)
    baseline_dist["method"] = "baseline"
    baseline_dist["lambda_penalty"] = float(lambda_penalty)
    rerank_dist = compute_topk_exposure_distribution(rerank_ranked, k=k)
    rerank_dist["method"] = "uncertainty_aware_rerank"
    rerank_dist["lambda_penalty"] = float(lambda_penalty)
    exposure_dist_df = pd.concat([baseline_dist, rerank_dist], ignore_index=True)
    save_table(exposure_dist_df, paths.tables_dir / "topk_exposure_distribution.csv")

    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_exp_name", type=str, required=True, help="Evidence posterior experiment name.")
    parser.add_argument("--input_path", type=str, default=None, help="Optional calibrated evidence posterior test jsonl.")
    parser.add_argument("--output_root", type=str, default="output-repaired", help="Output root.")
    parser.add_argument("--grid_exp_prefix", type=str, default=None, help="Optional prefix for per-lambda rerank experiments.")
    parser.add_argument("--summary_path", type=str, default=None, help="Optional grid summary CSV path.")
    parser.add_argument("--lambda_grid", type=str, default="0,0.1,0.2,0.5,1.0", help="Comma-separated lambda grid.")
    parser.add_argument("--score_column", type=str, default="repaired_confidence", help="Score column.")
    parser.add_argument("--uncertainty_column", type=str, default="evidence_uncertainty", help="Uncertainty column.")
    parser.add_argument("--method_variant", type=str, default="evidence_posterior_rerank_grid", help="Method variant tag.")
    parser.add_argument("--k", type=int, default=10, help="Top-K for ranking metrics.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    input_path = (
        Path(args.input_path)
        if args.input_path
        else Path(args.output_root) / args.base_exp_name / "calibrated" / "evidence_posterior_test.jsonl"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Rerank input not found: {input_path}")

    lambdas = parse_lambda_grid(args.lambda_grid)
    grid_prefix = args.grid_exp_prefix or f"{args.base_exp_name}_rerank"
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else Path(args.output_root) / "summary" / f"{grid_prefix}_grid.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    raw_df = load_jsonl(input_path)
    required_cols = [
        "user_id",
        "candidate_item_id",
        "label",
        "target_popularity_group",
        args.score_column,
        args.uncertainty_column,
    ]
    validate_input_columns(raw_df, required_cols)
    df = usable_rerank_rows(raw_df, score_column=args.score_column, uncertainty_column=args.uncertainty_column)
    if df.empty:
        raise ValueError("No usable rows remain after filtering parsed rows and required score columns.")

    print(f"[{args.base_exp_name}] Rerank grid input: {input_path}")
    print(f"[{args.base_exp_name}] Loaded rows={len(raw_df)} usable_rows={len(df)} lambdas={lambdas}")

    all_results: list[pd.DataFrame] = []
    for lambda_penalty in lambdas:
        exp_name = f"{grid_prefix}_{lambda_slug(lambda_penalty)}"
        print(f"[{exp_name}] Running lambda={lambda_penalty:g}")
        result_df = run_single_lambda(
            df=df,
            exp_name=exp_name,
            output_root=args.output_root,
            k=args.k,
            lambda_penalty=lambda_penalty,
            score_column=args.score_column,
            uncertainty_column=args.uncertainty_column,
            method_variant=args.method_variant,
        )
        all_results.append(result_df)

    summary_df = pd.concat(all_results, ignore_index=True)
    summary_df.insert(0, "base_exp_name", args.base_exp_name)
    summary_df.insert(1, "input_path", str(input_path))
    summary_df.insert(2, "usable_rows", int(len(df)))
    summary_df.insert(3, "total_rows", int(len(raw_df)))
    save_table(summary_df, summary_path)

    print(f"[{args.base_exp_name}] Rerank grid done.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

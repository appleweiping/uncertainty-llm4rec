from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from main_rerank import build_result_row, load_jsonl, save_jsonl, save_table
from main_rerank_grid import lambda_slug, parse_lambda_grid, usable_rerank_rows, validate_input_columns
from src.analysis.rerank_diagnostics import (
    add_evidence_risk,
    add_user_normalized_column,
    compute_rank_change_diagnostics,
)
from src.eval.bias_metrics import compute_topk_exposure_distribution
from src.methods.baseline_ranker import add_baseline_score, rank_by_score
from src.methods.uncertainty_reranker import rank_by_rerank_score
from src.utils.paths import ensure_exp_dirs


SETTINGS = {
    "A": {
        "base_score_col": "raw_confidence",
        "uncertainty_col": "minimal_evidence_uncertainty",
        "base_score_label": "raw_confidence",
        "uncertainty_label": "1_minus_repaired_confidence",
    },
    "B": {
        "base_score_col": "raw_confidence",
        "uncertainty_col": "evidence_risk",
        "base_score_label": "raw_confidence",
        "uncertainty_label": "evidence_risk",
    },
    "C": {
        "base_score_col": "minimal_repaired_confidence",
        "uncertainty_col": "evidence_risk",
        "base_score_label": "repaired_confidence",
        "uncertainty_label": "evidence_risk",
    },
}


def _prepare_setting_frame(
    df: pd.DataFrame,
    *,
    base_score_col: str,
    uncertainty_col: str,
    normalization: str,
) -> pd.DataFrame:
    out = add_user_normalized_column(
        df,
        base_score_col,
        "normalized_base_score",
        method=normalization,
    )
    out = add_user_normalized_column(
        out,
        uncertainty_col,
        "normalized_uncertainty",
        method=normalization,
    )
    return out


def _run_decoupled_lambda(
    *,
    df: pd.DataFrame,
    exp_name: str,
    output_root: str,
    setting: str,
    lambda_penalty: float,
    base_score_col: str,
    uncertainty_col: str,
    normalization: str,
    method_variant: str,
    k: int,
) -> pd.DataFrame:
    paths = ensure_exp_dirs(exp_name, output_root)
    prepared = _prepare_setting_frame(
        df,
        base_score_col=base_score_col,
        uncertainty_col=uncertainty_col,
        normalization=normalization,
    )

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
    save_jsonl(rerank_ranked, paths.reranked_dir / "decoupled_reranked.jsonl")

    baseline_row = build_result_row("baseline", baseline_ranked, k=k)
    rerank_row = build_result_row(
        "decoupled_uncertainty_aware_rerank",
        rerank_ranked,
        k=k,
        lambda_penalty=lambda_penalty,
    )
    results_df = pd.concat([baseline_row, rerank_row], ignore_index=True)
    diagnostics = compute_rank_change_diagnostics(
        baseline_ranked,
        rerank_ranked,
        base_score_col=base_score_col,
        uncertainty_col=uncertainty_col,
        lambda_penalty=lambda_penalty,
        setting=setting,
        normalization=normalization,
        k=k,
    )
    for key, value in diagnostics.items():
        if key not in results_df.columns:
            results_df[key] = value
    results_df["exp_name"] = exp_name
    results_df["method_variant"] = method_variant
    results_df["base_score_col"] = base_score_col
    results_df["uncertainty_col"] = uncertainty_col
    save_table(results_df, paths.tables_dir / "rerank_results.csv")
    save_table(results_df, paths.reranked_dir / "rerank_results.csv")

    baseline_dist = compute_topk_exposure_distribution(baseline_ranked, k=k)
    baseline_dist["method"] = "baseline"
    baseline_dist["lambda_penalty"] = float(lambda_penalty)
    rerank_dist = compute_topk_exposure_distribution(rerank_ranked, k=k)
    rerank_dist["method"] = "decoupled_uncertainty_aware_rerank"
    rerank_dist["lambda_penalty"] = float(lambda_penalty)
    save_table(
        pd.concat([baseline_dist, rerank_dist], ignore_index=True),
        paths.tables_dir / "topk_exposure_distribution.csv",
    )

    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_exp_name", type=str, required=True)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="output-repaired")
    parser.add_argument("--grid_exp_prefix", type=str, default=None)
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--lambda_grid", type=str, default="0,0.1,0.2,0.5,1.0")
    parser.add_argument("--settings", type=str, default="A,B,C")
    parser.add_argument("--normalization", type=str, default="minmax", choices=["minmax", "zscore", "none"])
    parser.add_argument("--evidence_risk_alpha", type=float, default=1.0 / 3.0)
    parser.add_argument("--evidence_risk_beta", type=float, default=1.0 / 3.0)
    parser.add_argument("--evidence_risk_gamma", type=float, default=1.0 / 3.0)
    parser.add_argument("--method_variant", type=str, default="evidence_posterior_decoupled_rerank_grid")
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
        raise FileNotFoundError(f"Decoupled rerank input not found: {input_path}")

    raw_df = load_jsonl(input_path)
    df_with_risk = add_evidence_risk(
        raw_df,
        output_col="evidence_risk",
        alpha=args.evidence_risk_alpha,
        beta=args.evidence_risk_beta,
        gamma=args.evidence_risk_gamma,
    )
    selected_settings = [item.strip().upper() for item in args.settings.split(",") if item.strip()]
    unknown = [setting for setting in selected_settings if setting not in SETTINGS]
    if unknown:
        raise ValueError(f"Unknown decoupled rerank settings: {unknown}")

    required_cols = ["user_id", "candidate_item_id", "label", "target_popularity_group"]
    for setting in selected_settings:
        required_cols.extend([SETTINGS[setting]["base_score_col"], SETTINGS[setting]["uncertainty_col"]])
    validate_input_columns(df_with_risk, sorted(set(required_cols)))

    # Use a permissive score pair for common parse filtering, then each setting normalizes its own columns.
    df = usable_rerank_rows(
        df_with_risk,
        score_column="raw_confidence",
        uncertainty_column="evidence_risk",
    )
    if df.empty:
        raise ValueError("No usable rows remain for decoupled rerank.")

    lambdas = parse_lambda_grid(args.lambda_grid)
    grid_prefix = args.grid_exp_prefix or f"{args.base_exp_name}_decoupled_rerank"
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else Path(args.output_root) / "summary" / f"{grid_prefix}_grid.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{args.base_exp_name}] Decoupled rerank input: {input_path}")
    print(
        f"[{args.base_exp_name}] usable_rows={len(df)} settings={selected_settings} "
        f"lambdas={lambdas} normalization={args.normalization}"
    )

    all_results: list[pd.DataFrame] = []
    for setting in selected_settings:
        spec = SETTINGS[setting]
        for lambda_penalty in lambdas:
            exp_name = f"{grid_prefix}_setting{setting}_{lambda_slug(lambda_penalty)}"
            print(f"[{exp_name}] Running setting={setting} lambda={lambda_penalty:g}")
            result_df = _run_decoupled_lambda(
                df=df,
                exp_name=exp_name,
                output_root=args.output_root,
                setting=setting,
                lambda_penalty=lambda_penalty,
                base_score_col=spec["base_score_col"],
                uncertainty_col=spec["uncertainty_col"],
                normalization=args.normalization,
                method_variant=args.method_variant,
                k=args.k,
            )
            result_df["setting"] = setting
            result_df["setting_base_score"] = spec["base_score_label"]
            result_df["setting_uncertainty"] = spec["uncertainty_label"]
            result_df["evidence_risk_alpha"] = args.evidence_risk_alpha
            result_df["evidence_risk_beta"] = args.evidence_risk_beta
            result_df["evidence_risk_gamma"] = args.evidence_risk_gamma
            all_results.append(result_df)

    summary_df = pd.concat(all_results, ignore_index=True)
    summary_df.insert(0, "base_exp_name", args.base_exp_name)
    summary_df.insert(1, "input_path", str(input_path))
    summary_df.insert(2, "usable_rows", int(len(df)))
    summary_df.insert(3, "total_rows", int(len(raw_df)))
    save_table(summary_df, summary_path)
    print(f"[{args.base_exp_name}] Decoupled rerank grid done.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

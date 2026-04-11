from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval.bias_metrics import compute_bias_metrics
from src.eval.calibration_metrics import compute_calibration_metrics, ensure_binary_columns
from src.eval.ranking_metrics import compute_ranking_metrics
from src.methods.baseline_ranker import add_baseline_score, rank_by_score
from src.methods.uncertainty_reranker import rerank_candidates
from src.uncertainty.estimators import (
    ensure_estimator_columns,
    get_available_estimators,
    merge_consistency_outputs,
)
from src.utils.paths import ensure_exp_dirs


def load_jsonl(path: str | Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _build_ranked_by_confidence(
    df: pd.DataFrame,
    confidence_col: str,
) -> pd.DataFrame:
    scored = add_baseline_score(df, score_col=confidence_col, output_col="baseline_score")
    return rank_by_score(scored, user_col="user_id", score_col="baseline_score", rank_col="rank")


def evaluate_estimator(
    df: pd.DataFrame,
    *,
    estimator_name: str,
    confidence_col: str,
    uncertainty_col: str,
    k: int,
    lambda_penalty: float,
) -> dict:
    eval_df = df[df[confidence_col].notna() & df[uncertainty_col].notna()].copy()
    if eval_df.empty:
        raise ValueError(
            f"No rows available for estimator '{estimator_name}' "
            f"after filtering on {confidence_col}/{uncertainty_col}."
        )

    metrics = compute_calibration_metrics(
        eval_df,
        confidence_col=confidence_col,
        target_col="is_correct",
    )

    rank_df = _build_ranked_by_confidence(eval_df, confidence_col=confidence_col)
    rank_metrics = compute_ranking_metrics(rank_df, k=k)
    rank_bias = compute_bias_metrics(rank_df, k=k)

    rerank_df = rerank_candidates(
        df=eval_df,
        user_col="user_id",
        confidence_col=confidence_col,
        uncertainty_col=uncertainty_col,
        lambda_penalty=lambda_penalty,
        score_col="final_score",
        rank_col="rank",
    )
    rerank_metrics = compute_ranking_metrics(rerank_df, k=k)
    rerank_bias = compute_bias_metrics(rerank_df, k=k)

    row = {
        "estimator": estimator_name,
        "confidence_col": confidence_col,
        "uncertainty_col": uncertainty_col,
        "lambda_penalty": float(lambda_penalty),
        "num_eval_samples": int(len(eval_df)),
        "num_eval_users": int(eval_df["user_id"].nunique()),
    }
    row.update({f"calib_{key}": value for key, value in metrics.items()})
    row.update({f"rank_{key}": value for key, value in rank_metrics.items()})
    row.update({f"rank_{key}": value for key, value in rank_bias.items()})
    row.update({f"rerank_{key}": value for key, value in rerank_metrics.items()})
    row.update({f"rerank_{key}": value for key, value in rerank_bias.items()})
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name.")
    parser.add_argument("--output_root", type=str, default="outputs", help="Output root.")
    parser.add_argument("--calibrated_path", type=str, default=None, help="Optional explicit path to test_calibrated.jsonl.")
    parser.add_argument("--consistency_path", type=str, default=None, help="Optional explicit path to self-consistency jsonl.")
    parser.add_argument("--k", type=int, default=10, help="Top-K for ranking evaluation.")
    parser.add_argument("--lambda_penalty", type=float, default=0.5, help="Lambda for rerank evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ensure_exp_dirs(args.exp_name, args.output_root)

    calibrated_path = (
        Path(args.calibrated_path)
        if args.calibrated_path is not None
        else paths.calibrated_dir / "test_calibrated.jsonl"
    )
    consistency_path = (
        Path(args.consistency_path)
        if args.consistency_path is not None
        else paths.root / "self_consistency" / "test_self_consistency.jsonl"
    )

    if not calibrated_path.exists():
        raise FileNotFoundError(f"Calibrated file not found: {calibrated_path}")

    print(f"[{args.exp_name}] Loading calibrated predictions from: {calibrated_path}")
    df = load_jsonl(calibrated_path)
    df = ensure_binary_columns(df)

    if consistency_path.exists():
        print(f"[{args.exp_name}] Loading self-consistency outputs from: {consistency_path}")
        consistency_df = load_jsonl(consistency_path)
        df = merge_consistency_outputs(df, consistency_df)
    else:
        print(f"[{args.exp_name}] Self-consistency file not found, skipping consistency-based estimators.")

    df = ensure_estimator_columns(df)
    estimators = get_available_estimators(df)
    if not estimators:
        raise ValueError("No available estimators found after loading inputs.")

    rows = []
    for estimator_name, cols in estimators.items():
        rows.append(
            evaluate_estimator(
                df=df,
                estimator_name=estimator_name,
                confidence_col=cols["confidence_col"],
                uncertainty_col=cols["uncertainty_col"],
                k=args.k,
                lambda_penalty=args.lambda_penalty,
            )
        )

    result_df = pd.DataFrame(rows)
    save_table(result_df, paths.tables_dir / "estimator_comparison.csv")

    print(f"[{args.exp_name}] Saved estimator comparison to: {paths.tables_dir / 'estimator_comparison.csv'}")


if __name__ == "__main__":
    main()

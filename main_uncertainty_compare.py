from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.eval.bias_metrics import compute_bias_metrics
from src.eval.calibration_metrics import compute_calibration_metrics, ensure_binary_columns
from src.eval.ranking_metrics import compute_ranking_metrics
from src.methods.baseline_ranker import add_baseline_score, rank_by_score
from src.methods.uncertainty_reranker import rerank_candidates
from src.uncertainty.estimators import (
    build_ranking_candidate_dataframe,
    build_ranking_proxy_dataframe,
    ensure_estimator_columns,
    get_available_estimators,
    merge_consistency_outputs,
)
from src.utils.paths import ensure_exp_dirs


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
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


def evaluate_pointwise_estimator(
    df: pd.DataFrame,
    *,
    estimator_name: str,
    confidence_col: str,
    uncertainty_col: str,
    k: int,
    lambda_penalty: float,
    fusion_alpha: float | None = None,
) -> dict[str, Any]:
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

    row: dict[str, Any] = {
        "task_type": "pointwise_yesno",
        "estimator": estimator_name,
        "confidence_col": confidence_col,
        "uncertainty_col": uncertainty_col,
        "lambda_penalty": float(lambda_penalty),
        "fusion_alpha": fusion_alpha if estimator_name == "fused" else pd.NA,
        "num_eval_samples": int(len(eval_df)),
        "num_eval_users": int(eval_df["user_id"].nunique()),
    }
    row.update({f"calib_{key}": value for key, value in metrics.items()})
    row.update({f"rank_{key}": value for key, value in rank_metrics.items()})
    row.update({f"rank_{key}": value for key, value in rank_bias.items()})
    row.update({f"rerank_{key}": value for key, value in rerank_metrics.items()})
    row.update({f"rerank_{key}": value for key, value in rerank_bias.items()})
    return row


def evaluate_ranking_proxy_estimator(
    proxy_df: pd.DataFrame,
    *,
    estimator_name: str,
    confidence_col: str,
    uncertainty_col: str,
    ranking_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eval_df = proxy_df[proxy_df[confidence_col].notna() & proxy_df[uncertainty_col].notna()].copy()
    if eval_df.empty:
        raise ValueError(
            f"No ranking proxy rows available for estimator '{estimator_name}' "
            f"after filtering on {confidence_col}/{uncertainty_col}."
        )

    calib_df = pd.DataFrame(
        {
            "recommend": ["yes"] * len(eval_df),
            "pred_label": [1] * len(eval_df),
            "label": eval_df["label"].astype(int),
            "confidence": eval_df[confidence_col].astype(float).clip(0.0, 1.0),
            "target_popularity_group": eval_df["target_popularity_group"].astype(str),
        }
    )
    calib_df = ensure_binary_columns(calib_df)
    metrics = compute_calibration_metrics(calib_df, confidence_col="confidence", target_col="is_correct")

    row: dict[str, Any] = {
        "task_type": "candidate_ranking",
        "estimator": estimator_name,
        "confidence_col": confidence_col,
        "uncertainty_col": uncertainty_col,
        "lambda_penalty": pd.NA,
        "fusion_alpha": pd.NA,
        "num_eval_samples": int(len(eval_df)),
        "num_eval_users": int(eval_df["user_id"].nunique()),
        "avg_top1_score": float(eval_df["top1_score"].mean()) if "top1_score" in eval_df.columns else float("nan"),
        "avg_top2_score": float(eval_df["top2_score"].mean()) if "top2_score" in eval_df.columns else float("nan"),
        "avg_score_margin": float(eval_df["score_margin"].mean()) if "score_margin" in eval_df.columns else float("nan"),
        "avg_score_entropy": float(eval_df["score_entropy"].mean()) if "score_entropy" in eval_df.columns else float("nan"),
    }
    row.update({f"calib_{key}": value for key, value in metrics.items()})

    if ranking_context:
        row.update({f"rank_{key}": value for key, value in ranking_context.items()})

    return row


def infer_task_type(df: pd.DataFrame, explicit_task_type: str) -> str:
    task_type = str(explicit_task_type).strip().lower()
    if task_type in {"pointwise_yesno", "candidate_ranking"}:
        return task_type

    if "task_type" in df.columns:
        values = {str(value).strip().lower() for value in df["task_type"].dropna().tolist() if str(value).strip()}
        if "candidate_ranking" in values:
            return "candidate_ranking"

    if {"ranked_item_ids", "candidate_scores", "selected_item_id"}.intersection(set(df.columns)):
        return "candidate_ranking"

    return "pointwise_yesno"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name.")
    parser.add_argument("--output_root", type=str, default="outputs", help="Output root.")
    parser.add_argument("--input_path", type=str, default=None, help="Optional primary input path.")
    parser.add_argument("--calibrated_path", type=str, default=None, help="Optional explicit path to test_calibrated.jsonl.")
    parser.add_argument("--consistency_path", type=str, default=None, help="Optional explicit path to self-consistency jsonl.")
    parser.add_argument("--ranking_proxy_path", type=str, default=None, help="Optional explicit path to ranking_proxy_rows.csv.")
    parser.add_argument("--task_type", type=str, default="auto", help="auto | pointwise_yesno | candidate_ranking")
    parser.add_argument("--k", type=int, default=10, help="Top-K for ranking evaluation.")
    parser.add_argument("--lambda_penalty", type=float, default=0.5, help="Lambda for rerank evaluation.")
    parser.add_argument(
        "--fused_alpha",
        type=float,
        default=0.5,
        help="Weight on calibrated verbalized confidence when forming fused uncertainty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ensure_exp_dirs(args.exp_name, args.output_root)

    resolved_input_path: Path | None = Path(args.input_path) if args.input_path is not None else None
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
    ranking_proxy_path = (
        Path(args.ranking_proxy_path)
        if args.ranking_proxy_path is not None
        else paths.tables_dir / "ranking_proxy_rows.csv"
    )

    if resolved_input_path is not None and resolved_input_path.exists():
        raw_df = load_table(resolved_input_path)
        task_type = infer_task_type(raw_df, explicit_task_type=args.task_type)
    elif args.task_type == "candidate_ranking":
        task_type = "candidate_ranking"
        raw_df = pd.DataFrame()
    else:
        task_type = "pointwise_yesno"
        raw_df = pd.DataFrame()

    rows: list[dict[str, Any]] = []

    if task_type == "pointwise_yesno":
        if not calibrated_path.exists():
            raise FileNotFoundError(f"Calibrated file not found: {calibrated_path}")

        print(f"[{args.exp_name}] Loading calibrated predictions from: {calibrated_path}")
        df = load_table(calibrated_path)
        df = ensure_binary_columns(df)

        if consistency_path.exists():
            print(f"[{args.exp_name}] Loading self-consistency outputs from: {consistency_path}")
            consistency_df = load_table(consistency_path)
            df = merge_consistency_outputs(df, consistency_df)
        else:
            print(f"[{args.exp_name}] Self-consistency file not found, skipping consistency-based estimators.")

        df = ensure_estimator_columns(df, fused_alpha=args.fused_alpha)
        estimators = get_available_estimators(df, fused_alpha=args.fused_alpha)
        if not estimators:
            raise ValueError("No available estimators found after loading pointwise inputs.")

        for estimator_name, cols in estimators.items():
            if cols.get("task_family") != "pointwise":
                continue
            rows.append(
                evaluate_pointwise_estimator(
                    df=df,
                    estimator_name=estimator_name,
                    confidence_col=str(cols["confidence_col"]),
                    uncertainty_col=str(cols["uncertainty_col"]),
                    k=args.k,
                    lambda_penalty=args.lambda_penalty,
                    fusion_alpha=cols.get("fusion_alpha") if isinstance(cols, dict) else None,
                )
            )
    else:
        if resolved_input_path is None:
            resolved_input_path = paths.predictions_dir / "test_ranking_raw.jsonl"
        if not resolved_input_path.exists():
            raise FileNotFoundError(f"Ranking prediction file not found: {resolved_input_path}")

        print(f"[{args.exp_name}] Loading ranking predictions from: {resolved_input_path}")
        raw_df = load_table(resolved_input_path)
        proxy_df = build_ranking_proxy_dataframe(raw_df)
        proxy_df = ensure_estimator_columns(proxy_df, fused_alpha=args.fused_alpha)
        estimators = get_available_estimators(proxy_df, fused_alpha=args.fused_alpha)
        if not estimators:
            raise ValueError("No available ranking proxy estimators found after loading inputs.")

        ranking_context = None
        if ranking_proxy_path.exists():
            print(f"[{args.exp_name}] Found existing ranking proxy rows at: {ranking_proxy_path}")
        candidate_df = None
        try:
            candidate_df = build_ranking_candidate_dataframe(raw_df)
        except Exception:
            candidate_df = None

        if candidate_df is not None and not candidate_df.empty:
            ranking_context = compute_ranking_metrics(candidate_df, k=args.k)
            ranking_context.update(compute_bias_metrics(candidate_df, k=args.k))

        for estimator_name, cols in estimators.items():
            if cols.get("task_family") != "ranking_proxy":
                continue
            rows.append(
                evaluate_ranking_proxy_estimator(
                    proxy_df=proxy_df,
                    estimator_name=estimator_name,
                    confidence_col=str(cols["confidence_col"]),
                    uncertainty_col=str(cols["uncertainty_col"]),
                    ranking_context=ranking_context,
                )
            )

    result_df = pd.DataFrame(rows)
    save_table(result_df, paths.tables_dir / "estimator_comparison.csv")
    print(f"[{args.exp_name}] Saved estimator comparison to: {paths.tables_dir / 'estimator_comparison.csv'}")


if __name__ == "__main__":
    main()

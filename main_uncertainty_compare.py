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
    DEFAULT_KEY_COLS,
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


def resolve_default_calibrated_path(exp_name: str, output_root: str) -> Path:
    root = Path(output_root) / exp_name / "calibrated"
    candidates = [
        root / "test_calibrated.jsonl",
        root / "evidence_posterior_test.jsonl",
        root / "evidence_posterior_minimal_test.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def default_evidence_source_paths(exp_name: str, output_root: str) -> dict[str, Path]:
    root = Path(output_root) / exp_name / "calibrated"
    return {
        "raw_calibrated": root / "raw_confidence_test_calibrated.jsonl",
        "evidence_minimal": root / "evidence_posterior_minimal_test.jsonl",
        "evidence_full": root / "evidence_posterior_full_test.jsonl",
    }


def merge_optional_source(
    base_df: pd.DataFrame,
    source_df: pd.DataFrame,
    columns_to_add: list[str],
    key_cols: list[str] | None = None,
) -> pd.DataFrame:
    key_cols = key_cols or DEFAULT_KEY_COLS
    shared_keys = [column for column in key_cols if column in base_df.columns and column in source_df.columns]
    if not shared_keys:
        raise ValueError("No shared key columns available for estimator source merge.")

    available_cols = [column for column in columns_to_add if column in source_df.columns]
    if not available_cols:
        return base_df

    merge_df = source_df[shared_keys + available_cols].drop_duplicates(subset=shared_keys)
    out = base_df.drop(columns=[column for column in available_cols if column in base_df.columns]).copy()
    return out.merge(merge_df, on=shared_keys, how="left")


def merge_evidence_sources(
    base_df: pd.DataFrame,
    *,
    raw_calibrated_path: Path | None = None,
    evidence_minimal_path: Path | None = None,
    evidence_full_path: Path | None = None,
) -> pd.DataFrame:
    out = base_df.copy()
    if raw_calibrated_path is not None and raw_calibrated_path.exists():
        out = merge_optional_source(
            out,
            load_jsonl(raw_calibrated_path),
            columns_to_add=["raw_calibrated_confidence"],
        )
    if evidence_minimal_path is not None and evidence_minimal_path.exists():
        out = merge_optional_source(
            out,
            load_jsonl(evidence_minimal_path),
            columns_to_add=["minimal_repaired_confidence", "minimal_evidence_uncertainty"],
        )
    if evidence_full_path is not None and evidence_full_path.exists():
        out = merge_optional_source(
            out,
            load_jsonl(evidence_full_path),
            columns_to_add=["full_repaired_confidence", "full_evidence_uncertainty"],
        )
    return out


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
    fusion_alpha: float | None = None,
) -> dict:
    eval_df = df[df[confidence_col].notna() & df[uncertainty_col].notna()].copy()
    if "parse_success" in eval_df.columns:
        eval_df = eval_df[eval_df["parse_success"].astype(bool)].copy()
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
    high_conf = eval_df[confidence_col].astype(float).clip(0.0, 1.0) >= 0.8
    wrong = eval_df["is_correct"].astype(int) == 0
    high_conf_error_rate = float((high_conf & wrong).sum() / high_conf.sum()) if int(high_conf.sum()) else 0.0

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
        "fusion_alpha": fusion_alpha if "fused" in estimator_name else pd.NA,
        "num_eval_samples": int(len(eval_df)),
        "num_eval_users": int(eval_df["user_id"].nunique()),
        "parse_success_rate": float(eval_df["parse_success"].mean()) if "parse_success" in eval_df.columns else pd.NA,
        "high_conf_threshold": 0.8,
        "high_conf_error_rate": high_conf_error_rate,
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
    parser.add_argument("--raw_calibrated_path", type=str, default=None, help="Optional raw confidence calibrated jsonl.")
    parser.add_argument("--evidence_minimal_path", type=str, default=None, help="Optional minimal evidence posterior jsonl.")
    parser.add_argument("--evidence_full_path", type=str, default=None, help="Optional full evidence posterior jsonl.")
    parser.add_argument("--consistency_path", type=str, default=None, help="Optional explicit path to self-consistency jsonl.")
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

    calibrated_path = (
        Path(args.calibrated_path)
        if args.calibrated_path is not None
        else resolve_default_calibrated_path(args.exp_name, args.output_root)
    )
    evidence_defaults = default_evidence_source_paths(args.exp_name, args.output_root)
    raw_calibrated_path = (
        Path(args.raw_calibrated_path)
        if args.raw_calibrated_path is not None
        else evidence_defaults["raw_calibrated"]
    )
    evidence_minimal_path = (
        Path(args.evidence_minimal_path)
        if args.evidence_minimal_path is not None
        else evidence_defaults["evidence_minimal"]
    )
    evidence_full_path = (
        Path(args.evidence_full_path)
        if args.evidence_full_path is not None
        else evidence_defaults["evidence_full"]
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
    df = merge_evidence_sources(
        df,
        raw_calibrated_path=raw_calibrated_path,
        evidence_minimal_path=evidence_minimal_path,
        evidence_full_path=evidence_full_path,
    )
    df = ensure_binary_columns(df)

    if consistency_path.exists():
        print(f"[{args.exp_name}] Loading self-consistency outputs from: {consistency_path}")
        consistency_df = load_jsonl(consistency_path)
        df = merge_consistency_outputs(df, consistency_df)
    else:
        print(f"[{args.exp_name}] Self-consistency file not found, skipping consistency-based estimators.")

    df = ensure_estimator_columns(df, fused_alpha=args.fused_alpha)
    estimators = get_available_estimators(df, fused_alpha=args.fused_alpha)
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
                fusion_alpha=cols.get("fusion_alpha") if isinstance(cols, dict) else None,
            )
        )

    result_df = pd.DataFrame(rows)
    save_table(result_df, paths.tables_dir / "estimator_comparison.csv")

    print(f"[{args.exp_name}] Saved estimator comparison to: {paths.tables_dir / 'estimator_comparison.csv'}")


if __name__ == "__main__":
    main()

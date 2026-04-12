from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main_uncertainty_compare import evaluate_estimator
from src.uncertainty.estimators import ensure_estimator_columns, merge_consistency_outputs
from src.utils.paths import ensure_exp_dirs


DEFAULT_ALPHAS = [0.2, 0.5, 0.8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="beauty_deepseek")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--alphas", type=float, nargs="*", default=DEFAULT_ALPHAS)
    parser.add_argument("--lambda_penalty", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required jsonl file: {path}")
    return pd.read_json(path, lines=True)


def select_brief_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_eval_samples": row.get("num_eval_samples"),
        "num_eval_users": row.get("num_eval_users"),
        "calibration_ece": row.get("calib_ece"),
        "calibration_brier_score": row.get("calib_brier_score"),
        "calibration_auroc": row.get("calib_auroc"),
        "rerank_ndcg_at_10": row.get("rerank_NDCG@10"),
        "rerank_mrr_at_10": row.get("rerank_MRR@10"),
        "rerank_head_exposure_ratio_at_10": row.get("rerank_head_exposure_ratio@10"),
        "rerank_long_tail_coverage_at_10": row.get("rerank_long_tail_coverage@10"),
    }


def build_reference_rows(
    df: pd.DataFrame,
    *,
    lambda_penalty: float,
    k: int,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for estimator_name, confidence_col, uncertainty_col in [
        (
            "verbalized_calibrated",
            "verbalized_calibrated_confidence",
            "verbalized_calibrated_uncertainty",
        ),
        ("consistency", "consistency_confidence", "consistency_uncertainty"),
    ]:
        result = evaluate_estimator(
            df=df,
            estimator_name=estimator_name,
            confidence_col=confidence_col,
            uncertainty_col=uncertainty_col,
            k=k,
            lambda_penalty=lambda_penalty,
            fusion_alpha=None,
        )
        row = {
            "exp_name": "beauty_deepseek",
            "model": "deepseek",
            "estimator": estimator_name,
            "fusion_alpha": pd.NA,
        }
        row.update(select_brief_metrics(result))
        refs.append(row)
    return refs


def build_fused_rows(
    base_df: pd.DataFrame,
    *,
    alphas: list[float],
    lambda_penalty: float,
    k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        df = ensure_estimator_columns(base_df, fused_alpha=alpha)
        result = evaluate_estimator(
            df=df,
            estimator_name="fused",
            confidence_col="fused_confidence",
            uncertainty_col="fused_uncertainty",
            k=k,
            lambda_penalty=lambda_penalty,
            fusion_alpha=alpha,
        )
        row = {
            "exp_name": "beauty_deepseek",
            "model": "deepseek",
            "estimator": "fused",
            "fusion_alpha": float(alpha),
        }
        row.update(select_brief_metrics(result))
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    paths = ensure_exp_dirs(args.exp_name, args.output_root)

    calibrated_path = paths.calibrated_dir / "test_calibrated.jsonl"
    consistency_path = paths.root / "self_consistency" / "test_self_consistency.jsonl"

    calibrated_df = load_jsonl(calibrated_path)
    consistency_df = load_jsonl(consistency_path)
    merged_df = merge_consistency_outputs(calibrated_df, consistency_df)
    merged_df = ensure_estimator_columns(merged_df, fused_alpha=0.5)

    rows: list[dict[str, Any]] = []
    rows.extend(
        build_reference_rows(
            merged_df,
            lambda_penalty=args.lambda_penalty,
            k=args.k,
        )
    )
    rows.extend(
        build_fused_rows(
            merged_df,
            alphas=[float(alpha) for alpha in args.alphas],
            lambda_penalty=args.lambda_penalty,
            k=args.k,
        )
    )

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(
        by=["estimator", "fusion_alpha"],
        na_position="first",
    ).reset_index(drop=True)

    out_path = Path(args.output_root).resolve() / "summary" / "beauty_fused_alpha_ablation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"Saved Beauty fused alpha ablation table to: {out_path}")


if __name__ == "__main__":
    main()

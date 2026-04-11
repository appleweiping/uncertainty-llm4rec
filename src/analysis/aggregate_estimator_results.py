from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs",
        help="Root directory containing experiment outputs.",
    )
    parser.add_argument(
        "--exp_names",
        type=str,
        nargs="*",
        default=None,
        help="Optional experiment folder names to aggregate. If omitted, auto-discover estimator comparison files.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Optional domain filter, e.g. beauty or movies.",
    )
    return parser.parse_args()


def resolve_estimator_table(exp_dir: Path) -> Path:
    candidates = [
        exp_dir / "estimator_comparison.csv",
        exp_dir / "tables" / "estimator_comparison.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing estimator_comparison.csv under {exp_dir}")


def discover_experiment_dirs(output_root: Path) -> list[Path]:
    experiment_dirs: list[Path] = []
    for child in sorted(output_root.iterdir()):
        if not child.is_dir():
            continue
        try:
            resolve_estimator_table(child)
        except FileNotFoundError:
            continue
        experiment_dirs.append(child)
    return experiment_dirs


def infer_domain_name(exp_name: str) -> str:
    tokens = [token.strip().lower() for token in exp_name.split("_") if token.strip()]
    if not tokens:
        return exp_name.lower()

    first = tokens[0]
    if first.startswith("movie"):
        return "movies"
    if first.startswith("book"):
        return "books"
    if first.startswith("electronic"):
        return "electronics"
    return first


def infer_model_name(exp_name: str) -> str:
    tokens = [token.strip().lower() for token in exp_name.split("_") if token.strip()]
    known_models = ["deepseek", "qwen", "kimi", "doubao", "glm", "gpt", "openai", "local"]
    for token in reversed(tokens):
        if token in known_models:
            return token
    return "unknown"


def normalize_metric_columns(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "calib_accuracy": row.get("calib_accuracy"),
        "calib_avg_confidence": row.get("calib_avg_confidence"),
        "calib_brier_score": row.get("calib_brier_score"),
        "calib_ece": row.get("calib_ece"),
        "calib_auroc": row.get("calib_auroc"),
        "rank_ndcg_at_10": row.get("rank_NDCG@10"),
        "rank_mrr_at_10": row.get("rank_MRR@10"),
        "rank_head_exposure_ratio_at_10": row.get("rank_head_exposure_ratio@10"),
        "rank_tail_exposure_ratio_at_10": row.get("rank_tail_exposure_ratio@10"),
        "rank_long_tail_coverage_at_10": row.get("rank_long_tail_coverage@10"),
        "rerank_ndcg_at_10": row.get("rerank_NDCG@10"),
        "rerank_mrr_at_10": row.get("rerank_MRR@10"),
        "rerank_head_exposure_ratio_at_10": row.get("rerank_head_exposure_ratio@10"),
        "rerank_tail_exposure_ratio_at_10": row.get("rerank_tail_exposure_ratio@10"),
        "rerank_long_tail_coverage_at_10": row.get("rerank_long_tail_coverage@10"),
    }


def aggregate_experiment(exp_dir: Path) -> list[dict[str, Any]]:
    exp_name = exp_dir.name
    domain = infer_domain_name(exp_name)
    model = infer_model_name(exp_name)

    df = pd.read_csv(resolve_estimator_table(exp_dir))
    if df.empty:
        return []

    rows: list[dict[str, Any]] = []
    for _, record in df.iterrows():
        row = {
            "exp_name": exp_name,
            "domain": domain,
            "model": model,
            "estimator": record.get("estimator"),
            "confidence_col": record.get("confidence_col"),
            "uncertainty_col": record.get("uncertainty_col"),
            "lambda": record.get("lambda_penalty", 0.0),
            "num_eval_samples": record.get("num_eval_samples"),
            "num_eval_users": record.get("num_eval_users"),
        }
        row.update(normalize_metric_columns(record.to_dict()))
        rows.append(row)
    return rows


def build_beauty_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    beauty_df = summary_df[summary_df["domain"].astype(str).str.lower() == "beauty"].copy()
    if beauty_df.empty:
        return beauty_df

    columns = [
        "domain",
        "model",
        "estimator",
        "lambda",
        "num_eval_samples",
        "calib_ece",
        "calib_brier_score",
        "calib_auroc",
        "rank_ndcg_at_10",
        "rank_mrr_at_10",
        "rank_head_exposure_ratio_at_10",
        "rank_long_tail_coverage_at_10",
        "rerank_ndcg_at_10",
        "rerank_mrr_at_10",
        "rerank_head_exposure_ratio_at_10",
        "rerank_long_tail_coverage_at_10",
    ]
    existing = [column for column in columns if column in beauty_df.columns]
    return beauty_df[existing].copy()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    if args.exp_names:
        exp_dirs = [(output_root / exp_name).resolve() for exp_name in args.exp_names]
    else:
        exp_dirs = discover_experiment_dirs(output_root)

    rows: list[dict[str, Any]] = []
    for exp_dir in exp_dirs:
        rows.extend(aggregate_experiment(exp_dir))

    if not rows:
        raise ValueError(f"No estimator comparison rows found under {output_root}")

    summary_df = pd.DataFrame(rows)
    if args.domain:
        summary_df = summary_df[
            summary_df["domain"].astype(str).str.lower() == args.domain.strip().lower()
        ].copy()

    summary_df = summary_df.sort_values(
        ["domain", "model", "estimator", "lambda"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    summary_dir = output_root / "summary"
    ensure_dir(summary_dir)

    estimator_output_path = summary_dir / "estimator_results.csv"
    beauty_output_path = summary_dir / "beauty_estimator_results.csv"

    summary_df.to_csv(estimator_output_path, index=False)
    build_beauty_summary(summary_df).to_csv(beauty_output_path, index=False)

    print(f"Aggregated {len(summary_df)} estimator rows.")
    print(f"Saved estimator comparison summary to: {estimator_output_path}")
    print(f"Saved Beauty-focused estimator summary to: {beauty_output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import ensure_dir


REQUIRED_FILENAMES = [
    "diagnostic_metrics.csv",
    "calibration_comparison.csv",
    "rerank_results.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--exp_names", type=str, nargs="*", default=None)
    parser.add_argument("--domain", type=str, default=None)
    return parser.parse_args()


def resolve_table_path(exp_dir: Path, filename: str) -> Path:
    candidates = [
        exp_dir / filename,
        exp_dir / "tables" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing required result file `{filename}` under {exp_dir}")


def discover_experiment_dirs(output_root: Path) -> list[Path]:
    experiment_dirs: list[Path] = []
    for child in sorted(output_root.iterdir()):
        if not child.is_dir() or child.name in {"summary", "robustness", "clean", "noisy"}:
            continue
        if should_skip_experiment_dir(child.name):
            continue
        try:
            for filename in REQUIRED_FILENAMES:
                resolve_table_path(child, filename)
        except FileNotFoundError:
            continue
        experiment_dirs.append(child)
    return experiment_dirs


def should_skip_experiment_dir(exp_name: str) -> bool:
    normalized = exp_name.strip().lower()
    return "_noisy" in normalized or re.search(r"_rep\d+$", normalized) is not None


def infer_domain_name(exp_name: str) -> str:
    tokens = [token.strip().lower() for token in exp_name.split("_") if token.strip()]
    ignore_tokens = {
        "deepseek",
        "qwen",
        "kimi",
        "doubao",
        "glm",
        "gpt",
        "openai",
        "local",
        "small",
        "mini",
        "large",
        "clean",
        "noisy",
        "robustness",
        "calibration",
        "rerank",
        "reranking",
    }
    domain_tokens = [token for token in tokens if token not in ignore_tokens and not token.startswith("lambda")]
    if not domain_tokens:
        return exp_name.lower()

    domain = domain_tokens[0]
    if domain.startswith("movie"):
        return "movies"
    if domain.startswith("book"):
        return "books"
    if domain.startswith("electronic"):
        return "electronics"
    return domain


def infer_model_name(exp_name: str) -> str:
    tokens = [token.strip().lower() for token in exp_name.split("_") if token.strip()]
    known_models = ["deepseek", "qwen", "kimi", "doubao", "glm", "gpt", "openai", "local"]
    for token in reversed(tokens):
        if token in known_models:
            return token
    return "unknown"


def load_diagnostic_metrics(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {
        "diagnostic_num_samples": row.get("num_samples"),
        "diagnostic_accuracy": row.get("accuracy"),
        "diagnostic_avg_confidence": row.get("avg_confidence"),
        "diagnostic_brier_score": row.get("brier_score"),
        "diagnostic_ece": row.get("ece"),
        "diagnostic_auroc": row.get("auroc"),
    }


def load_calibration_metrics(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    if df.empty:
        return {}
    test_df = df[df["split"].astype(str).str.lower() == "test"].copy()
    if test_df.empty:
        return {}

    metrics: dict[str, Any] = {}
    for _, row in test_df.iterrows():
        metric = str(row["metric"]).strip().lower()
        safe_name = metric.replace("@", "_at_").replace("-", "_")
        metrics[f"calibration_test_{safe_name}_before"] = row.get("before")
        metrics[f"calibration_test_{safe_name}_after"] = row.get("after")
    return metrics


def load_rerank_metrics(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    if df.empty:
        return {}

    baseline_df = df[df["method"].astype(str).str.lower() == "baseline"].copy()
    rerank_df = df[df["method"].astype(str).str.lower() == "uncertainty_aware_rerank"].copy()
    if rerank_df.empty and baseline_df.empty:
        return {}

    baseline_row = baseline_df.iloc[0].to_dict() if not baseline_df.empty else {}
    rerank_row = rerank_df.iloc[0].to_dict() if not rerank_df.empty else baseline_row
    lambda_value = rerank_row.get("lambda_penalty", 0.0) if rerank_row else 0.0

    return {
        "lambda": lambda_value if pd.notna(lambda_value) else 0.0,
        "baseline_hr_at_10": baseline_row.get("HR@10"),
        "baseline_ndcg_at_10": baseline_row.get("NDCG@10"),
        "baseline_mrr_at_10": baseline_row.get("MRR@10"),
        "baseline_num_users": baseline_row.get("num_users"),
        "baseline_num_samples": baseline_row.get("num_samples"),
        "baseline_head_exposure_ratio_at_10": baseline_row.get("head_exposure_ratio@10"),
        "baseline_tail_exposure_ratio_at_10": baseline_row.get("tail_exposure_ratio@10"),
        "baseline_long_tail_coverage_at_10": baseline_row.get("long_tail_coverage@10"),
        "rerank_hr_at_10": rerank_row.get("HR@10"),
        "rerank_ndcg_at_10": rerank_row.get("NDCG@10"),
        "rerank_mrr_at_10": rerank_row.get("MRR@10"),
        "rerank_num_users": rerank_row.get("num_users"),
        "rerank_num_samples": rerank_row.get("num_samples"),
        "rerank_head_exposure_ratio_at_10": rerank_row.get("head_exposure_ratio@10"),
        "rerank_tail_exposure_ratio_at_10": rerank_row.get("tail_exposure_ratio@10"),
        "rerank_long_tail_coverage_at_10": rerank_row.get("long_tail_coverage@10"),
    }


def aggregate_experiment(exp_dir: Path) -> dict[str, Any]:
    exp_name = exp_dir.name
    row: dict[str, Any] = {
        "exp_name": exp_name,
        "domain": infer_domain_name(exp_name),
        "model": infer_model_name(exp_name),
    }
    row.update(load_diagnostic_metrics(resolve_table_path(exp_dir, "diagnostic_metrics.csv")))
    row.update(load_calibration_metrics(resolve_table_path(exp_dir, "calibration_comparison.csv")))
    row.update(load_rerank_metrics(resolve_table_path(exp_dir, "rerank_results.csv")))
    return row


def build_domain_model_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "diagnostic_accuracy",
        "diagnostic_ece",
        "diagnostic_brier_score",
        "calibration_test_ece_before",
        "calibration_test_ece_after",
        "calibration_test_brier_score_before",
        "calibration_test_brier_score_after",
        "baseline_hr_at_10",
        "baseline_ndcg_at_10",
        "baseline_mrr_at_10",
        "baseline_head_exposure_ratio_at_10",
        "baseline_tail_exposure_ratio_at_10",
        "baseline_long_tail_coverage_at_10",
        "rerank_hr_at_10",
        "rerank_ndcg_at_10",
        "rerank_mrr_at_10",
        "rerank_head_exposure_ratio_at_10",
        "rerank_tail_exposure_ratio_at_10",
        "rerank_long_tail_coverage_at_10",
    ]
    existing_metrics = [col for col in metric_columns if col in summary_df.columns]
    if not existing_metrics:
        return pd.DataFrame()

    grouped = (
        summary_df.groupby(["domain", "model"], dropna=False)[existing_metrics]
        .mean(numeric_only=True)
        .reset_index()
    )
    return grouped.sort_values(["domain", "model"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    if args.exp_names:
        exp_dirs = [output_root / exp_name for exp_name in args.exp_names]
    else:
        exp_dirs = discover_experiment_dirs(output_root)

    if not exp_dirs:
        raise ValueError(f"No experiment folders with required files were found under {output_root}")

    rows = [aggregate_experiment(exp_dir) for exp_dir in exp_dirs]
    summary_df = pd.DataFrame(rows)
    if args.domain:
        summary_df = summary_df[
            summary_df["domain"].astype(str).str.lower() == args.domain.strip().lower()
        ].copy()

    summary_df = summary_df.sort_values(["domain", "model", "lambda"], ascending=[True, True, True]).reset_index(drop=True)

    summary_dir = output_root / "summary"
    ensure_dir(summary_dir)

    model_output_path = summary_dir / "model_results.csv"
    grouped_output_path = summary_dir / "domain_model_summary.csv"

    summary_df.to_csv(model_output_path, index=False)
    build_domain_model_summary(summary_df).to_csv(grouped_output_path, index=False)

    print(f"Aggregated {len(summary_df)} model rows.")
    print(f"Saved model comparison summary to: {model_output_path}")
    print(f"Saved domain-model summary to: {grouped_output_path}")


if __name__ == "__main__":
    main()

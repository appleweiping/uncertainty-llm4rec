from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
import re

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

KNOWN_MODELS = {
    "deepseek",
    "qwen",
    "kimi",
    "doubao",
    "glm",
    "gpt",
    "openai",
    "local",
}


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
        help="Optional experiment folder names to aggregate. If omitted, auto-discover folders with required result files.",
    )
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


def infer_domain_name(exp_name: str) -> str:
    tokens = [token.strip().lower() for token in exp_name.split("_") if token.strip()]
    ignore_tokens = {
        *KNOWN_MODELS,
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
    for token in reversed(tokens):
        if token in KNOWN_MODELS:
            return token
    return "unknown"


def discover_experiment_dirs(output_root: Path) -> list[Path]:
    experiment_dirs: list[Path] = []
    for child in sorted(output_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in {"summary", "robustness", "clean", "noisy"}:
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


def load_diagnostic_metrics(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Diagnostic metrics file is empty: {path}")

    row = df.iloc[0].to_dict()
    return {
        "diagnostic_num_samples": row.get("num_samples"),
        "diagnostic_accuracy": row.get("accuracy"),
        "diagnostic_avg_confidence": row.get("avg_confidence"),
        "diagnostic_avg_correctness": row.get("avg_correctness"),
        "diagnostic_brier_score": row.get("brier_score"),
        "diagnostic_ece": row.get("ece"),
        "diagnostic_mce": row.get("mce"),
        "diagnostic_auroc": row.get("auroc"),
    }


def load_calibration_metrics(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Calibration comparison file is empty: {path}")

    test_df = df[df["split"].astype(str).str.lower() == "test"].copy()
    if test_df.empty:
        raise ValueError(f"No test split rows found in calibration comparison file: {path}")

    metrics: dict[str, Any] = {}
    for _, row in test_df.iterrows():
        metric_name = str(row["metric"]).strip().lower()
        safe_name = metric_name.replace("@", "_at_").replace("-", "_")
        metrics[f"calibration_test_{safe_name}_before"] = row.get("before")
        metrics[f"calibration_test_{safe_name}_after"] = row.get("after")
    return metrics


def _extract_method_row(df: pd.DataFrame, method_name: str) -> pd.Series | None:
    method_df = df[df["method"].astype(str).str.lower() == method_name.lower()].copy()
    if method_df.empty:
        return None
    return method_df.iloc[0]


def load_rerank_metrics(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Rerank results file is empty: {path}")

    baseline_row = _extract_method_row(df, "baseline")
    rerank_row = _extract_method_row(df, "uncertainty_aware_rerank")

    if rerank_row is None and baseline_row is None:
        raise ValueError(f"No usable rerank rows found in: {path}")

    if rerank_row is None:
        rerank_row = baseline_row

    lambda_value = rerank_row.get("lambda_penalty") if rerank_row is not None else None
    if pd.isna(lambda_value):
        lambda_value = 0.0

    result: dict[str, Any] = {
        "lambda": float(lambda_value),
    }

    metric_columns = [
        "HR@10",
        "NDCG@10",
        "MRR@10",
        "num_users",
        "num_samples",
        "head_exposure_ratio@10",
        "tail_exposure_ratio@10",
        "long_tail_coverage@10",
        "topk_total_items@10",
    ]

    def add_prefixed_metrics(prefix: str, row: pd.Series | None) -> None:
        if row is None:
            for column in metric_columns:
                safe_name = column.lower().replace("@", "_at_")
                result[f"{prefix}_{safe_name}"] = pd.NA
            return

        for column in metric_columns:
            safe_name = column.lower().replace("@", "_at_")
            result[f"{prefix}_{safe_name}"] = row.get(column)

    add_prefixed_metrics("baseline", baseline_row)
    add_prefixed_metrics("rerank", rerank_row)
    return result


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


def build_weekly_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "domain",
        "model",
        "exp_name",
        "lambda",
        "diagnostic_accuracy",
        "diagnostic_ece",
        "diagnostic_brier_score",
        "calibration_test_ece_before",
        "calibration_test_ece_after",
        "calibration_test_brier_score_before",
        "calibration_test_brier_score_after",
        "rerank_hr_at_10",
        "rerank_ndcg_at_10",
        "rerank_mrr_at_10",
        "rerank_head_exposure_ratio_at_10",
        "rerank_tail_exposure_ratio_at_10",
        "rerank_long_tail_coverage_at_10",
    ]
    existing = [column for column in columns if column in summary_df.columns]
    return summary_df[existing].copy()


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
    summary_df = summary_df.sort_values(["domain", "lambda"], ascending=[True, True]).reset_index(drop=True)

    summary_dir = output_root / "summary"
    ensure_dir(summary_dir)
    rerank_output_path = summary_dir / "rerank_ablation.csv"
    weekly_output_path = summary_dir / "weekly_summary.csv"
    final_results_path = summary_dir / "final_results.csv"

    summary_df.to_csv(rerank_output_path, index=False)
    build_weekly_summary(summary_df).to_csv(weekly_output_path, index=False)
    summary_df.to_csv(final_results_path, index=False)

    print(f"Aggregated {len(summary_df)} experiment rows.")
    print(f"Saved rerank ablation to: {rerank_output_path}")
    print(f"Saved weekly summary to: {weekly_output_path}")
    print(f"Saved final results to: {final_results_path}")


if __name__ == "__main__":
    main()

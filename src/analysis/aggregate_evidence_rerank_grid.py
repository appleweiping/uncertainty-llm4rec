from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


METRIC_COLUMNS = [
    "HR@10",
    "NDCG@10",
    "MRR@10",
    "head_exposure_ratio@10",
    "tail_exposure_ratio@10",
    "long_tail_coverage@10",
]


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def load_grid_summary(path: str | Path) -> pd.DataFrame:
    grid_path = Path(path)
    if not grid_path.exists():
        raise FileNotFoundError(f"Rerank grid summary not found: {grid_path}")
    df = pd.read_csv(grid_path)
    if df.empty:
        raise ValueError(f"Rerank grid summary is empty: {grid_path}")
    required = {"method", "exp_name", "base_exp_name", "lambda_penalty"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns from rerank grid summary: {missing}")
    return df


def aggregate_grid(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for exp_name, group in df.groupby("exp_name", dropna=False):
        baseline_rows = group[group["method"].astype(str).str.lower() == "baseline"].copy()
        rerank_rows = group[group["method"].astype(str).str.lower() == "uncertainty_aware_rerank"].copy()
        if rerank_rows.empty:
            continue

        baseline_row = baseline_rows.iloc[0].to_dict() if not baseline_rows.empty else {}
        rerank_row = rerank_rows.iloc[0].to_dict()
        lambda_penalty = _safe_float(rerank_row.get("lambda_penalty"))

        row: dict[str, Any] = {
            "base_exp_name": rerank_row.get("base_exp_name"),
            "rerank_exp_name": exp_name,
            "method_variant": rerank_row.get("method_variant"),
            "score_column": rerank_row.get("score_column"),
            "uncertainty_column": rerank_row.get("uncertainty_column"),
            "lambda_penalty": lambda_penalty,
            "input_path": rerank_row.get("input_path"),
            "total_rows": rerank_row.get("total_rows"),
            "usable_rows": rerank_row.get("usable_rows"),
            "num_users": rerank_row.get("num_users"),
            "num_samples": rerank_row.get("num_samples"),
        }

        for metric in METRIC_COLUMNS:
            safe_name = metric.lower().replace("@", "_at_")
            baseline_value = _safe_float(baseline_row.get(metric))
            rerank_value = _safe_float(rerank_row.get(metric))
            row[f"baseline_{safe_name}"] = baseline_value
            row[f"rerank_{safe_name}"] = rerank_value
            row[f"delta_{safe_name}"] = rerank_value - baseline_value

        rows.append(row)

    if not rows:
        raise ValueError("No uncertainty_aware_rerank rows found in grid summary.")

    out = pd.DataFrame(rows)
    return out.sort_values("lambda_penalty", na_position="first").reset_index(drop=True)


def select_best_lambda(aggregated_df: pd.DataFrame, metric_col: str = "rerank_ndcg_at_10") -> pd.Series:
    if metric_col not in aggregated_df.columns:
        raise ValueError(f"Metric column not found: {metric_col}")
    work = aggregated_df.dropna(subset=[metric_col]).copy()
    if work.empty:
        return aggregated_df.iloc[0]
    return work.sort_values([metric_col, "lambda_penalty"], ascending=[False, True]).iloc[0]


def write_markdown_conclusion(
    aggregated_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Beauty Evidence Rerank Ablation",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best = select_best_lambda(aggregated_df)
    baseline_ndcg = _safe_float(best.get("baseline_ndcg_at_10"))
    best_ndcg = _safe_float(best.get("rerank_ndcg_at_10"))
    best_delta = _safe_float(best.get("delta_ndcg_at_10"))
    best_lambda = _safe_float(best.get("lambda_penalty"))

    lines = [
        f"# {title}",
        "",
        "This table is a Day6 aggregation artifact for the evidence posterior rerank grid. It does not claim a final ranking gain by itself; it records whether repaired confidence can be injected into the existing week3 decision pipeline without breaking utility or exposure accounting.",
        "",
        "## Best Lambda By NDCG@10",
        "",
        f"- Best lambda: `{best_lambda:g}`",
        f"- Baseline NDCG@10: `{baseline_ndcg:.6f}`",
        f"- Rerank NDCG@10: `{best_ndcg:.6f}`",
        f"- Delta NDCG@10: `{best_delta:.6f}`",
        "",
        "## Interpretation Guardrail",
        "",
    ]

    if best_delta > 0:
        lines.append("The current grid shows a positive NDCG@10 delta for at least one lambda. This is promising, but it should only be treated as a stable downstream gain after full Beauty valid/test evidence predictions are available and the same trend survives larger-scale rerank evaluation.")
    elif best_delta == 0:
        lines.append("The current grid is utility-neutral at the best lambda. This is still a useful engineering result: evidence posterior uncertainty can pass through the rerank interface and preserve the baseline ranking behavior, but it should not be claimed as a ranking improvement yet.")
    else:
        lines.append("The current grid does not improve NDCG@10 at the best observed lambda. The correct conclusion is conservative: the repaired confidence is decision-compatible, but downstream gains require larger candidate sets, more stable full Beauty evidence predictions, or lambda tuning.")

    table_columns = [
        "lambda_penalty",
        "baseline_ndcg_at_10",
        "rerank_ndcg_at_10",
        "delta_ndcg_at_10",
        "baseline_mrr_at_10",
        "rerank_mrr_at_10",
        "delta_mrr_at_10",
        "baseline_head_exposure_ratio_at_10",
        "rerank_head_exposure_ratio_at_10",
        "delta_head_exposure_ratio_at_10",
        "baseline_long_tail_coverage_at_10",
        "rerank_long_tail_coverage_at_10",
        "delta_long_tail_coverage_at_10",
    ]
    existing_columns = [column for column in table_columns if column in aggregated_df.columns]
    lines.extend(
        [
            "",
            "## Aggregated Rows",
            "",
            dataframe_to_markdown(aggregated_df[existing_columns]),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [format_markdown_value(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def save_aggregated_outputs(
    grid_summary_path: str | Path,
    output_csv_path: str | Path,
    output_md_path: str | Path | None = None,
) -> pd.DataFrame:
    df = load_grid_summary(grid_summary_path)
    aggregated_df = aggregate_grid(df)

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated_df.to_csv(output_csv_path, index=False)

    if output_md_path is not None:
        write_markdown_conclusion(aggregated_df, output_md_path)

    return aggregated_df

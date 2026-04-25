from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_csv(path: str | Path, label: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_csv(path)


def _best_decoupled_rows(decoupled_df: pd.DataFrame) -> pd.DataFrame:
    rerank_df = decoupled_df[decoupled_df["method"] == "decoupled_uncertainty_aware_rerank"].copy()
    if rerank_df.empty:
        return rerank_df
    return (
        rerank_df.sort_values(["setting", "NDCG@10", "MRR@10"], ascending=[True, False, False])
        .groupby("setting", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def build_comparison(
    monotonic_grid_df: pd.DataFrame,
    monotonic_diagnostics_df: pd.DataFrame,
    decoupled_grid_df: pd.DataFrame,
) -> pd.DataFrame:
    monotonic_rerank = monotonic_grid_df[monotonic_grid_df["method"] == "uncertainty_aware_rerank"].copy()
    monotonic_best = monotonic_rerank.sort_values(["NDCG@10", "MRR@10"], ascending=[False, False]).head(1).copy()
    monotonic_diag = monotonic_diagnostics_df.copy()
    monotonic_noop = bool(monotonic_diag["rerank_is_noop"].all()) if "rerank_is_noop" in monotonic_diag else False
    monotonic_rank_change = float(monotonic_diag["rank_change_rate"].mean()) if "rank_change_rate" in monotonic_diag else float("nan")

    rows: list[dict] = []
    if not monotonic_best.empty:
        row = monotonic_best.iloc[0].to_dict()
        rows.append(
            {
                "family": "monotonic_original",
                "setting": "monotonic",
                "lambda_penalty": row.get("lambda_penalty"),
                "base_score": row.get("score_column"),
                "uncertainty": row.get("uncertainty_column"),
                "normalization": "none",
                "HR@10": row.get("HR@10"),
                "NDCG@10": row.get("NDCG@10"),
                "MRR@10": row.get("MRR@10"),
                "head_exposure_ratio@10": row.get("head_exposure_ratio@10"),
                "tail_exposure_ratio@10": row.get("tail_exposure_ratio@10"),
                "long_tail_coverage@10": row.get("long_tail_coverage@10"),
                "rank_change_rate": monotonic_rank_change,
                "rerank_is_noop": monotonic_noop,
                "diagnosis": (
                    "No-op: uncertainty is 1 - repaired_confidence, so final_score is a "
                    "monotonic affine transform of repaired_confidence for non-negative lambda."
                ),
            }
        )

    for _, row in _best_decoupled_rows(decoupled_grid_df).iterrows():
        rows.append(
            {
                "family": "decoupled",
                "setting": row.get("setting"),
                "lambda_penalty": row.get("lambda_penalty"),
                "base_score": row.get("setting_base_score"),
                "uncertainty": row.get("setting_uncertainty"),
                "normalization": row.get("normalization"),
                "HR@10": row.get("HR@10"),
                "NDCG@10": row.get("NDCG@10"),
                "MRR@10": row.get("MRR@10"),
                "head_exposure_ratio@10": row.get("head_exposure_ratio@10"),
                "tail_exposure_ratio@10": row.get("tail_exposure_ratio@10"),
                "long_tail_coverage@10": row.get("long_tail_coverage@10"),
                "rank_change_rate": row.get("rank_change_rate"),
                "rerank_is_noop": bool(row.get("rerank_is_noop", False)),
                "diagnosis": "Decoupled base score and uncertainty; lambda can change within-user order.",
            }
        )

    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame) -> str:
    columns = [
        "family",
        "setting",
        "lambda_penalty",
        "base_score",
        "uncertainty",
        "NDCG@10",
        "MRR@10",
        "rank_change_rate",
        "rerank_is_noop",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_markdown(comparison_df: pd.DataFrame) -> str:
    lines = [
        "# Beauty Evidence Rerank Comparison",
        "",
        "The original Day5 grid is a diagnostic ablation, not evidence against evidence posterior. "
        "Because it used repaired_confidence as the base score and 1 - repaired_confidence as the penalty, "
        "the final score is a monotonic affine transform for non-negative lambda and cannot change ranking order.",
        "",
        _markdown_table(comparison_df),
        "",
        "The decoupled grid separates relevance-like base scores from risk-like uncertainty signals. "
        "This is the decision experiment to carry forward into Day6-Day10.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monotonic_grid_path", type=str, required=True)
    parser.add_argument("--monotonic_diagnostics_path", type=str, required=True)
    parser.add_argument("--decoupled_grid_path", type=str, required=True)
    parser.add_argument("--output_csv_path", type=str, required=True)
    parser.add_argument("--output_md_path", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_df = build_comparison(
        _read_csv(args.monotonic_grid_path, "Monotonic grid"),
        _read_csv(args.monotonic_diagnostics_path, "Monotonic diagnostics"),
        _read_csv(args.decoupled_grid_path, "Decoupled grid"),
    )
    output_csv_path = Path(args.output_csv_path)
    output_md_path = Path(args.output_md_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_csv_path, index=False)
    output_md_path.write_text(build_markdown(comparison_df), encoding="utf-8")
    print(f"Saved rerank comparison CSV to: {output_csv_path}")
    print(f"Saved rerank comparison markdown to: {output_md_path}")


if __name__ == "__main__":
    main()

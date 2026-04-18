from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def collect_baseline_runs(output_root: str | Path = "outputs/baselines") -> list[dict[str, str]]:
    root = Path(output_root)
    if not root.exists():
        return []

    runs: list[dict[str, str]] = []
    for baseline_dir in root.iterdir():
        if not baseline_dir.is_dir() or baseline_dir.name == "summary":
            continue
        for exp_dir in baseline_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            runs.append(
                {
                    "baseline_name": baseline_dir.name,
                    "exp_name": exp_dir.name,
                    "root": str(exp_dir),
                }
            )
    return sorted(runs, key=lambda row: (row["baseline_name"], row["exp_name"]))


def build_baseline_summary_tables(output_root: str | Path = "outputs/baselines") -> dict[str, pd.DataFrame]:
    runs = collect_baseline_runs(output_root=output_root)

    metrics_rows: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, Any]] = []
    paper_rows: list[dict[str, Any]] = []

    for run in runs:
        root = Path(run["root"])
        metrics_df = _safe_read_csv(root / "metrics" / "ranking_metrics.csv")
        proxy_df = _safe_read_csv(root / "proxy" / "baseline_proxy_summary.csv")

        if not metrics_df.empty:
            metrics_row = metrics_df.iloc[0].to_dict()
            metrics_row.update(
                {
                    "baseline_name": run["baseline_name"],
                    "exp_name": run["exp_name"],
                }
            )
            metrics_rows.append(metrics_row)

        if not proxy_df.empty:
            proxy_row = proxy_df.iloc[0].to_dict()
            proxy_row.update(
                {
                    "baseline_name": run["baseline_name"],
                    "exp_name": run["exp_name"],
                }
            )
            proxy_rows.append(proxy_row)

        if not metrics_df.empty and not proxy_df.empty:
            metrics_row = metrics_df.iloc[0].to_dict()
            proxy_row = proxy_df.iloc[0].to_dict()
            paper_rows.append(
                {
                    "baseline_name": run["baseline_name"],
                    "exp_name": run["exp_name"],
                    "NDCG@10": metrics_row.get("NDCG@10"),
                    "HR@10": metrics_row.get("HR@10"),
                    "Recall@10": metrics_row.get("Recall@10"),
                    "MRR@10": metrics_row.get("MRR@10"),
                    "avg_top1_score": proxy_row.get("avg_top1_score"),
                    "avg_score_margin": proxy_row.get("avg_score_margin"),
                    "avg_score_entropy": proxy_row.get("avg_score_entropy"),
                    "avg_proxy_confidence": proxy_row.get("avg_proxy_confidence"),
                    "top1_accuracy": proxy_row.get("top1_accuracy"),
                    "ECE": proxy_row.get("ECE", proxy_row.get("ece")),
                    "Brier": proxy_row.get("Brier", proxy_row.get("brier_score")),
                    "wrong_high_conf_fraction": proxy_row.get("wrong_high_conf_fraction"),
                }
            )

    metrics_summary = pd.DataFrame(metrics_rows)
    proxy_summary = pd.DataFrame(proxy_rows)
    paper_table = pd.DataFrame(paper_rows)
    if not paper_table.empty:
        order = {"cove": 0, "slmrec": 1, "llm_esr": 2}
        paper_table["sort_key"] = paper_table["baseline_name"].map(order).fillna(999)
        paper_table = paper_table.sort_values(["sort_key", "exp_name"]).drop(columns=["sort_key"]).reset_index(drop=True)

    return {
        "baseline_metrics_summary": metrics_summary,
        "baseline_proxy_summary": proxy_summary,
        "baseline_paper_table": paper_table,
    }

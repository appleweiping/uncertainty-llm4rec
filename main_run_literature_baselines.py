from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.baselines.literature_pairwise_baseline import PAIRWISE_BASELINE_BUILDERS, PAIRWISE_BASELINE_NOTES
from src.baselines.literature_rank_baseline import BASELINE_BUILDERS, BASELINE_NOTES
from src.eval.preference_metrics import (
    build_pairwise_eval_frame,
    compute_pairwise_metrics,
    compute_preference_confidence_bins,
)
from src.eval.ranking_task_metrics import (
    build_ranking_eval_frame,
    compute_ranking_exposure_distribution,
    compute_ranking_task_metrics,
)
from src.utils.exp_io import load_yaml
from src.utils.io import ensure_dir, load_jsonl, save_jsonl


DEFAULT_RANK_BASELINES = [
    "candidate_order_rank",
    "popularity_prior_rank",
    "longtail_prior_rank",
    "history_overlap_rank",
]

DEFAULT_PAIRWISE_BASELINES = [
    "history_overlap_pairwise",
    "popularity_prior_pairwise",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run first-round Week7 literature-aligned baselines.")
    parser.add_argument("--config", type=str, default=None, help="Optional baseline config YAML.")
    parser.add_argument("--domain", type=str, default="beauty")
    parser.add_argument("--model", type=str, default="llama31_8b_instruct_local")
    parser.add_argument("--model_family", type=str, default="local_hf_base_only")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--rank_input_path", type=str, default="data/processed/amazon_beauty/ranking_test.jsonl")
    parser.add_argument("--pairwise_input_path", type=str, default="data/processed/amazon_beauty/pairwise_coverage_test.jsonl")
    parser.add_argument("--rank_baselines", nargs="*", default=None)
    parser.add_argument("--pairwise_baselines", nargs="*", default=None)
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--summary_path", type=str, default="outputs/summary/week7_day3_literature_baseline_summary.csv")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=100)
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if args.config:
        cfg = load_yaml(args.config)

    return {
        "domain": cfg.get("domain", args.domain),
        "model": cfg.get("model", args.model),
        "model_family": cfg.get("model_family", args.model_family),
        "adapter_path": cfg.get("adapter_path", args.adapter_path),
        "rank_input_path": cfg.get("rank_input_path", args.rank_input_path),
        "pairwise_input_path": cfg.get("pairwise_input_path", args.pairwise_input_path),
        "rank_baselines": args.rank_baselines or cfg.get("rank_baselines") or DEFAULT_RANK_BASELINES,
        "pairwise_baselines": args.pairwise_baselines or cfg.get("pairwise_baselines") or DEFAULT_PAIRWISE_BASELINES,
        "output_root": cfg.get("output_root", args.output_root),
        "summary_path": cfg.get("summary_path", args.summary_path),
        "k": int(cfg.get("k", args.k)),
        "max_samples": int(cfg.get("max_samples", args.max_samples)),
    }


def run_rank_baselines(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    input_path = Path(str(cfg["rank_input_path"]))
    if not input_path.exists():
        raise FileNotFoundError(f"Rank baseline input not found: {input_path}")
    samples = load_jsonl(input_path)[: int(cfg["max_samples"])]
    rows: list[dict[str, Any]] = []

    for baseline_name in cfg["rank_baselines"]:
        if baseline_name not in BASELINE_BUILDERS:
            raise ValueError(f"Unsupported ranking baseline: {baseline_name}")
        predictions = BASELINE_BUILDERS[baseline_name](samples, k=int(cfg["k"]))
        output_dir = Path(str(cfg["output_root"])) / "baselines" / "literature" / f"{cfg['domain']}_{cfg['model']}_{baseline_name}"
        prediction_dir = output_dir / "predictions"
        table_dir = output_dir / "tables"
        ensure_dir(prediction_dir)
        ensure_dir(table_dir)

        prediction_path = prediction_dir / "rank_predictions.jsonl"
        save_jsonl(predictions, prediction_path)
        eval_df = build_ranking_eval_frame(pd.DataFrame(predictions))
        metrics = compute_ranking_task_metrics(eval_df, k=int(cfg["k"]))
        exposure_df = compute_ranking_exposure_distribution(eval_df, k=int(cfg["k"]))
        metrics_path = table_dir / "ranking_metrics.csv"
        exposure_path = table_dir / "ranking_exposure_distribution.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        exposure_df.to_csv(exposure_path, index=False)

        rows.append(
            {
                "task": "candidate_ranking",
                "baseline_family": "literature_aligned_rank",
                "baseline_name": baseline_name,
                "domain": cfg["domain"],
                "model": cfg["model"],
                "model_family": cfg["model_family"],
                "adapter_path": cfg.get("adapter_path") or "",
                "samples": metrics.get("sample_count"),
                "HR@10": metrics.get("HR@10"),
                "NDCG@10": metrics.get("NDCG@10"),
                "MRR": metrics.get("MRR"),
                "pairwise_accuracy": pd.NA,
                "parse_success_rate": metrics.get("parse_success_rate"),
                "output_dir": str(output_dir),
                "prediction_path": str(prediction_path),
                "metrics_path": str(metrics_path),
                "schema_aligned": True,
                "notes": BASELINE_NOTES.get(baseline_name, ""),
            }
        )
    return rows


def run_pairwise_baselines(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    input_path = Path(str(cfg["pairwise_input_path"]))
    if not input_path.exists():
        raise FileNotFoundError(f"Pairwise baseline input not found: {input_path}")
    samples = load_jsonl(input_path)[: int(cfg["max_samples"])]
    rows: list[dict[str, Any]] = []

    for baseline_name in cfg["pairwise_baselines"]:
        if baseline_name not in PAIRWISE_BASELINE_BUILDERS:
            raise ValueError(f"Unsupported pairwise baseline: {baseline_name}")
        predictions = PAIRWISE_BASELINE_BUILDERS[baseline_name](samples)
        output_dir = Path(str(cfg["output_root"])) / "baselines" / "literature" / f"{cfg['domain']}_{cfg['model']}_{baseline_name}"
        prediction_dir = output_dir / "predictions"
        table_dir = output_dir / "tables"
        ensure_dir(prediction_dir)
        ensure_dir(table_dir)

        prediction_path = prediction_dir / "pairwise_predictions.jsonl"
        save_jsonl(predictions, prediction_path)
        eval_df = build_pairwise_eval_frame(pd.DataFrame(predictions))
        metrics = compute_pairwise_metrics(eval_df)
        bins_df = compute_preference_confidence_bins(eval_df)
        metrics_path = table_dir / "pairwise_metrics.csv"
        bins_path = table_dir / "preference_confidence_bins.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        bins_df.to_csv(bins_path, index=False)

        rows.append(
            {
                "task": "pairwise_preference",
                "baseline_family": "literature_aligned_pairwise",
                "baseline_name": baseline_name,
                "domain": cfg["domain"],
                "model": cfg["model"],
                "model_family": cfg["model_family"],
                "adapter_path": cfg.get("adapter_path") or "",
                "samples": metrics.get("sample_count"),
                "HR@10": pd.NA,
                "NDCG@10": pd.NA,
                "MRR": pd.NA,
                "pairwise_accuracy": metrics.get("pairwise_accuracy"),
                "parse_success_rate": metrics.get("parse_success_rate"),
                "output_dir": str(output_dir),
                "prediction_path": str(prediction_path),
                "metrics_path": str(metrics_path),
                "schema_aligned": True,
                "notes": PAIRWISE_BASELINE_NOTES.get(baseline_name, ""),
            }
        )
    return rows


def main() -> None:
    cfg = merge_config(parse_args())
    rows = run_rank_baselines(cfg) + run_pairwise_baselines(cfg)
    summary_path = Path(str(cfg["summary_path"]))
    ensure_dir(summary_path.parent)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Saved Week7 Day3 literature baseline summary to: {summary_path}")


if __name__ == "__main__":
    main()


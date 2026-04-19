from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.baselines.literature_rank_baselines import build_popularity_prior_predictions
from src.eval.ranking_task_metrics import (
    build_ranking_eval_frame,
    compute_ranking_exposure_distribution,
    compute_ranking_task_metrics,
)
from src.utils.io import ensure_dir, load_jsonl, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_name", type=str, default="popularity_prior_rank")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="week6_magic7_popularity_prior")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--status_path", type=str, default="outputs/summary/week6_magic7_literature_baseline_status.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Baseline input not found: {input_path}")

    samples = load_jsonl(input_path)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    if args.baseline_name != "popularity_prior_rank":
        raise ValueError(f"Unsupported minimal literature baseline: {args.baseline_name}")

    output_dir = Path(args.output_root) / "baselines" / "literature" / args.exp_name
    prediction_dir = output_dir / "predictions"
    table_dir = output_dir / "tables"
    ensure_dir(prediction_dir)
    ensure_dir(table_dir)

    predictions = build_popularity_prior_predictions(samples, k=args.k)
    prediction_path = prediction_dir / "rank_predictions.jsonl"
    save_jsonl(predictions, prediction_path)

    eval_df = build_ranking_eval_frame(pd.DataFrame(predictions))
    metrics = compute_ranking_task_metrics(eval_df, k=args.k)
    exposure_df = compute_ranking_exposure_distribution(eval_df, k=args.k)
    metrics_path = table_dir / "ranking_metrics.csv"
    exposure_path = table_dir / "ranking_exposure_distribution.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    exposure_df.to_csv(exposure_path, index=False)

    status_path = Path(args.status_path)
    ensure_dir(status_path.parent)
    status_df = pd.DataFrame(
        [
            {
                "baseline_name": args.baseline_name,
                "task": "candidate_ranking",
                "is_runnable": True,
                "schema_aligned": True,
                "input_path": str(input_path),
                "output_dir": str(output_dir),
                "prediction_path": str(prediction_path),
                "metrics_path": str(metrics_path),
                "sample_count": metrics.get("sample_count"),
                "HR@10": metrics.get("HR@10"),
                "NDCG@10": metrics.get("NDCG@10"),
                "MRR": metrics.get("MRR"),
                "notes": "Minimal literature-aligned popularity prior baseline; intended as a schema bridge, not the final strong baseline set.",
            }
        ]
    )
    status_df.to_csv(status_path, index=False)

    print(f"Saved baseline predictions to: {prediction_path}")
    print(f"Saved baseline metrics to: {metrics_path}")
    print(f"Saved baseline status to: {status_path}")


if __name__ == "__main__":
    main()


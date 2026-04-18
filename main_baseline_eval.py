from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.baseline.adapters import CoVEAdapter, SLMRecAdapter
from src.baseline.eval import build_ranked_rows, build_score_rows, compute_nh_nr_metrics
from src.baseline.io import ensure_baseline_dirs, load_grouped_candidate_samples, save_jsonl_records, save_table
from src.utils.reproducibility import set_global_seed


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Optional baseline eval config path.")
    parser.add_argument("--baseline_name", type=str, default=None, help="Baseline name: cove or slmrec.")
    parser.add_argument("--exp_name", type=str, default=None, help="Baseline experiment name.")
    parser.add_argument("--input_path", type=str, default=None, help="Grouped candidate input path.")
    parser.add_argument("--train_path", type=str, default=None, help="Optional grouped candidate train path for fit.")
    parser.add_argument("--output_root", type=str, default=None, help="Baseline output root.")
    parser.add_argument("--k", type=int, default=None, help="Top-k cutoff.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample cap.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed.")
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if args.config:
        cfg = load_yaml(args.config)
    return {
        "baseline_name": args.baseline_name if args.baseline_name is not None else cfg.get("baseline_name"),
        "exp_name": args.exp_name if args.exp_name is not None else cfg.get("exp_name", "baseline_eval"),
        "input_path": args.input_path if args.input_path is not None else cfg.get("input_path"),
        "train_path": args.train_path if args.train_path is not None else cfg.get("train_path"),
        "output_root": args.output_root if args.output_root is not None else cfg.get("output_root", "outputs/baselines"),
        "k": args.k if args.k is not None else cfg.get("k", 10),
        "max_samples": args.max_samples if args.max_samples is not None else cfg.get("max_samples"),
        "seed": args.seed if args.seed is not None else cfg.get("seed"),
    }


def build_adapter(name: str):
    normalized = str(name or "").strip().lower()
    if normalized == "cove":
        return CoVEAdapter()
    if normalized == "slmrec":
        return SLMRecAdapter()
    raise ValueError(f"Unsupported baseline_name: {name}")


def main() -> None:
    args = parse_args()
    cfg = merge_config(args)

    baseline_name = str(cfg["baseline_name"] or "").strip().lower()
    if not baseline_name:
        raise ValueError("baseline_name is required.")
    if not cfg["input_path"]:
        raise ValueError("input_path is required.")

    if cfg["seed"] is not None:
        set_global_seed(int(cfg["seed"]))

    paths = ensure_baseline_dirs(
        baseline_name=baseline_name,
        exp_name=str(cfg["exp_name"]),
        output_root=cfg["output_root"],
    )

    grouped_samples = load_grouped_candidate_samples(cfg["input_path"], max_samples=cfg["max_samples"])
    train_samples = (
        load_grouped_candidate_samples(cfg["train_path"], max_samples=cfg["max_samples"])
        if cfg["train_path"]
        else grouped_samples
    )

    adapter = build_adapter(baseline_name)
    adapter.fit(train_samples)
    predictions = adapter.predict_groups(grouped_samples)

    predictions_path = paths.predictions_dir / "baseline_predictions.jsonl"
    save_jsonl_records(predictions, predictions_path)

    ranked_rows = build_ranked_rows(grouped_samples, predictions)
    score_rows = build_score_rows(ranked_rows)
    metrics = compute_nh_nr_metrics(ranked_rows, k=int(cfg["k"]))

    save_table(ranked_rows, paths.predictions_dir / "rank_rows.csv")
    save_table(score_rows, paths.predictions_dir / "score_rows.csv")
    save_table(pd.DataFrame([metrics]), paths.metrics_dir / "ranking_metrics.csv")
    save_table(pd.DataFrame([cfg]), paths.logs_dir / "run_config.csv")

    print(f"[{baseline_name}:{cfg['exp_name']}] grouped samples: {len(grouped_samples)}")
    print(f"[{baseline_name}:{cfg['exp_name']}] predictions: {predictions_path}")
    print(f"[{baseline_name}:{cfg['exp_name']}] metrics: {paths.metrics_dir / 'ranking_metrics.csv'}")


if __name__ == "__main__":
    main()

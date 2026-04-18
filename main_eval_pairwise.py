from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval.preference_metrics import (
    build_pairwise_eval_frame,
    compute_pairwise_metrics,
    compute_preference_confidence_bins,
)
from src.utils.exp_io import load_jsonl
from src.utils.paths import ensure_exp_dirs
from src.utils.reproducibility import set_global_seed


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_summary_dict(summary: dict, path: str | Path) -> None:
    save_table(pd.DataFrame([summary]), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name, e.g. beauty_qwen_pairwise")
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Optional manual prediction path. Default: outputs/{exp_name}/predictions/pairwise_predictions.jsonl",
    )
    parser.add_argument("--output_root", type=str, default="outputs", help="Output root directory.")
    parser.add_argument("--n_bins", type=int, default=10, help="Confidence bin count.")
    parser.add_argument("--seed", type=int, default=None, help="Optional global random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    paths = ensure_exp_dirs(args.exp_name, args.output_root)
    input_path = Path(args.input_path) if args.input_path else paths.predictions_dir / "pairwise_predictions.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(f"Pairwise prediction file not found: {input_path}")

    print(f"[{args.exp_name}] Loading pairwise predictions from: {input_path}")
    raw_records = load_jsonl(input_path)
    raw_df = pd.DataFrame(raw_records)
    eval_df = build_pairwise_eval_frame(raw_df)

    print(f"[{args.exp_name}] Loaded {len(eval_df)} pairwise samples.")
    if args.seed is not None:
        print(f"[{args.exp_name}] Seed: {args.seed}")

    metrics = compute_pairwise_metrics(eval_df)
    bins_df = compute_preference_confidence_bins(eval_df, n_bins=args.n_bins)

    save_summary_dict(metrics, paths.tables_dir / "pairwise_metrics.csv")
    save_table(bins_df, paths.tables_dir / "preference_confidence_bins.csv")
    save_table(eval_df, paths.tables_dir / "pairwise_eval_records.csv")

    print(f"[{args.exp_name}] Pairwise evaluation done.")
    print(f"[{args.exp_name}] Tables saved to: {paths.tables_dir}")


if __name__ == "__main__":
    main()

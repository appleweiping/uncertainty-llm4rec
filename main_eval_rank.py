from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval.ranking_task_metrics import (
    build_ranking_eval_frame,
    compute_ranking_exposure_distribution,
    compute_ranking_task_metrics,
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
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name, e.g. beauty_qwen_rank")
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Optional manual prediction path. Default: outputs/{exp_name}/predictions/rank_predictions.jsonl",
    )
    parser.add_argument("--output_root", type=str, default="outputs", help="Output root directory.")
    parser.add_argument("--k", type=int, default=10, help="Top-k cutoff for ranking metrics.")
    parser.add_argument("--seed", type=int, default=None, help="Optional global random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    paths = ensure_exp_dirs(args.exp_name, args.output_root)
    input_path = Path(args.input_path) if args.input_path else paths.predictions_dir / "rank_predictions.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(f"Ranking prediction file not found: {input_path}")

    print(f"[{args.exp_name}] Loading ranking predictions from: {input_path}")
    raw_records = load_jsonl(input_path)
    raw_df = pd.DataFrame(raw_records)
    eval_df = build_ranking_eval_frame(raw_df)

    print(f"[{args.exp_name}] Loaded {len(eval_df)} ranking samples.")
    if args.seed is not None:
        print(f"[{args.exp_name}] Seed: {args.seed}")

    metrics = compute_ranking_task_metrics(eval_df, k=args.k)
    exposure_df = compute_ranking_exposure_distribution(eval_df, k=args.k)

    save_summary_dict(metrics, paths.tables_dir / "ranking_metrics.csv")
    save_table(exposure_df, paths.tables_dir / "ranking_exposure_distribution.csv")
    save_table(eval_df, paths.tables_dir / "ranking_eval_records.csv")

    print(f"[{args.exp_name}] Ranking evaluation done.")
    print(f"[{args.exp_name}] Tables saved to: {paths.tables_dir}")


if __name__ == "__main__":
    main()

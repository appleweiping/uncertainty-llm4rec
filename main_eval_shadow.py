from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.shadow.eval import (
    compute_shadow_diagnostic_metrics,
    compute_shadow_score_summary,
    prepare_shadow_dataframe,
    save_table,
    shadow_reliability_dataframe,
)
from src.utils.paths import ensure_exp_dirs
from src.utils.reproducibility import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--input_path", default=None)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--score_col", default="shadow_score")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    paths = ensure_exp_dirs(args.exp_name, args.output_root)
    input_path = Path(args.input_path) if args.input_path else paths.predictions_dir / "test_raw.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {input_path}")

    raw_df = pd.read_json(input_path, lines=True)
    df = prepare_shadow_dataframe(raw_df, score_col=args.score_col, threshold=args.threshold)
    metrics = compute_shadow_diagnostic_metrics(df, n_bins=args.n_bins)
    summary = compute_shadow_score_summary(df)
    reliability = shadow_reliability_dataframe(df, n_bins=args.n_bins)

    save_table(pd.DataFrame([metrics]), paths.tables_dir / "diagnostic_metrics.csv")
    save_table(pd.DataFrame([summary]), paths.tables_dir / "shadow_score_summary.csv")
    save_table(reliability, paths.tables_dir / "reliability_bins.csv")
    print(f"[{args.exp_name}] Shadow evaluation done.")
    print(f"[{args.exp_name}] Tables saved to: {paths.tables_dir}")


if __name__ == "__main__":
    main()

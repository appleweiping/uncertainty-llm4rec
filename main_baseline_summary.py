from __future__ import annotations

import argparse
from pathlib import Path

from src.baseline.summary import build_baseline_summary_tables
from src.baseline.io import save_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="outputs/baselines", help="Baseline output root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_root = Path(args.output_root) / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    tables = build_baseline_summary_tables(output_root=args.output_root)
    for name, df in tables.items():
        save_table(df, summary_root / f"{name}.csv")
        print(f"Saved {name}: {summary_root / f'{name}.csv'}")


if __name__ == "__main__":
    main()

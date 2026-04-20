from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.aggregate_framework_compare import (
    FRAMEWORK_COMPARE_COLUMNS,
    build_framework_compare_rows,
    write_framework_compare,
)
from src.utils.exp_io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Week7.5 framework config path.")
    parser.add_argument("--output_path", type=str, default=None, help="Optional compare CSV override.")
    parser.add_argument("--init_only", action="store_true", help="Only initialize the compare schema if the file is missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    default_output_path = (
        config.get("summary", {}) or {}
    ).get("framework_compare_path", "outputs/summary/week7_5_framework_compare.csv")
    output_path = Path(args.output_path or default_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.init_only and output_path.exists():
        print(f"Week7.5 framework compare already exists: {output_path}")
        return

    rows = build_framework_compare_rows(args.config)
    if not rows:
        rows = []
    write_framework_compare(rows, output_path)
    print(f"Saved Week7.5 framework compare to: {output_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()

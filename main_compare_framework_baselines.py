from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.aggregate_framework_baseline_bridge import (
    build_framework_baseline_bridge,
    write_framework_baseline_bridge,
    write_framework_baseline_bridge_markdown,
)
from src.utils.exp_io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge the Week7.5 framework compare into the baseline matrix schema.")
    parser.add_argument("--config", type=str, required=True, help="Week7.5 framework config path.")
    parser.add_argument("--baseline_matrix_path", type=str, default=None, help="Optional Week7 baseline matrix override.")
    parser.add_argument("--framework_compare_path", type=str, default=None, help="Optional framework compare CSV override.")
    parser.add_argument("--output_path", type=str, default=None, help="Optional bridged baseline matrix output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    summary_cfg = config.get("summary", {}) or {}
    baseline_matrix_path = Path(
        args.baseline_matrix_path or summary_cfg.get("baseline_matrix_source_path", "outputs/summary/week7_day4_baseline_matrix.csv")
    )
    framework_compare_path = Path(
        args.framework_compare_path or summary_cfg.get("framework_compare_path", "outputs/summary/week7_5_framework_compare.csv")
    )
    output_path = Path(
        args.output_path or summary_cfg.get("framework_baseline_bridge_path", "outputs/summary/week7_5_baseline_matrix.csv")
    )
    markdown_path = Path(
        summary_cfg.get("framework_baseline_bridge_markdown_path", "outputs/summary/week7_5_baseline_matrix.md")
    )

    rows = build_framework_baseline_bridge(
        baseline_matrix_path=baseline_matrix_path,
        framework_compare_path=framework_compare_path,
    )
    output_path = write_framework_baseline_bridge(rows, output_path)
    write_framework_baseline_bridge_markdown(rows, markdown_path)
    print(f"Saved Week7.5 framework baseline bridge to: {output_path}")
    print(f"Saved Week7.5 framework baseline bridge markdown to: {markdown_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()

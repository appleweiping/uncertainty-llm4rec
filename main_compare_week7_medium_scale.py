from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.aggregate_week7_medium_scale import (
    DEFAULT_BASELINE_MATRIX_PATH,
    DEFAULT_BATCH_STATUS_PATH,
    DEFAULT_OUTPUT_PATH,
    build_medium_scale_summary,
    write_medium_scale_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Week7 medium-scale handoff summary.")
    parser.add_argument("--batch_status_path", type=str, default=str(DEFAULT_BATCH_STATUS_PATH))
    parser.add_argument("--baseline_matrix_path", type=str, default=str(DEFAULT_BASELINE_MATRIX_PATH))
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df = build_medium_scale_summary(
        batch_status_path=Path(args.batch_status_path),
        baseline_matrix_path=Path(args.baseline_matrix_path),
    )
    output_path = write_medium_scale_summary(summary_df, Path(args.output_path))
    print(f"Saved Week7 medium-scale summary to: {output_path}")
    print(f"Rows: {len(summary_df)}")


if __name__ == "__main__":
    main()

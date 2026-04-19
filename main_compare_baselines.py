from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.aggregate_baseline_results import (
    DEFAULT_DECISION_BASELINE_PATH,
    DEFAULT_ESTIMATOR_COMPARE_PATH,
    DEFAULT_LITERATURE_BASELINE_PATH,
    DEFAULT_OUTPUT_PATH,
    build_baseline_matrix,
    write_baseline_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Week7 unified baseline matrix.")
    parser.add_argument("--estimator_compare_path", type=str, default=str(DEFAULT_ESTIMATOR_COMPARE_PATH))
    parser.add_argument("--decision_baseline_path", type=str, default=str(DEFAULT_DECISION_BASELINE_PATH))
    parser.add_argument("--literature_baseline_path", type=str, default=str(DEFAULT_LITERATURE_BASELINE_PATH))
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = build_baseline_matrix(
        estimator_compare_path=Path(args.estimator_compare_path),
        decision_baseline_path=Path(args.decision_baseline_path),
        literature_baseline_path=Path(args.literature_baseline_path),
    )
    output_path = write_baseline_matrix(matrix, Path(args.output_path))
    print(f"Saved unified baseline matrix to: {output_path}")
    print(f"Rows: {len(matrix)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from src.training.srpd_dataset import build_srpd_rank_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SRPD ranking teacher data from structured-risk artifacts.")
    parser.add_argument("--config", required=True, help="Path to an SRPD data config YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_srpd_rank_data(args.config)
    print(f"Saved SRPD {summary['srpd_stage']} train data to: {summary['output_train_path']}")
    print(f"Saved SRPD {summary['srpd_stage']} valid data to: {summary['output_valid_path']}")
    print(f"Matched rows: {summary['matched_rows']} / {summary['teacher_rows']}")
    print(f"Status: {summary['status']}")


if __name__ == "__main__":
    main()

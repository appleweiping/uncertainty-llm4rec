from __future__ import annotations

import argparse

from src.training import run_lora_rank_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="LoRA framework config path.")
    parser.add_argument("--dry_run", action="store_true", help="Only materialize the training skeleton and previews.")
    parser.add_argument(
        "--startup_check",
        action="store_true",
        help="Validate paths, sample loading, prompt construction, and output directories without entering training.",
    )
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional train-split override.")
    parser.add_argument("--max_valid_samples", type=int, default=None, help="Optional valid-split override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_lora_rank_training(
        args.config,
        dry_run=args.dry_run,
        startup_check=args.startup_check,
        max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples,
    )
    if args.startup_check:
        print(f"Saved Week7.5 startup check to: {summary['startup_check_path']}")
        print(f"Saved dataset preview to: {summary['dataset_preview_path']}")
        return
    print(f"Saved Week7.5 ranking LoRA training summary to: {summary['logs_dir']}/training_summary.csv")
    print(f"Adapter output directory: {summary['adapter_output_dir']}")


if __name__ == "__main__":
    main()

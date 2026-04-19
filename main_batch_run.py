from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils.exp_io import load_yaml
from src.utils.exp_launcher import build_experiment_row, launch_experiment
from src.utils.exp_registry import failed_exp_names, write_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or register a structured experiment batch.")
    parser.add_argument("--batch_config", type=str, default=None, help="Batch YAML with experiment specs.")
    parser.add_argument("--configs", type=str, nargs="*", default=None, help="Legacy direct config list.")
    parser.add_argument("--status_path", type=str, default=None, help="CSV registry output path.")
    parser.add_argument("--status_only", action="store_true", help="Only write registry status rows; do not run.")
    parser.add_argument("--dry_run", action="store_true", help="Build commands and validate inputs without running.")
    parser.add_argument("--run", action="store_true", help="Actually launch experiments. Without this flag the launcher is dry-run.")
    parser.add_argument("--only_failed", action="store_true", help="Only rerun experiments marked failed in the existing registry.")
    parser.add_argument("--python_bin", type=str, default=None, help="Python executable to use on the server.")
    parser.add_argument("--max_experiments", type=int, default=None, help="Optional cap for smoke-testing a batch list.")
    return parser.parse_args()


def load_batch(args: argparse.Namespace) -> tuple[str, str, str, str, list[dict[str, Any]]]:
    if args.batch_config:
        batch_path = Path(args.batch_config)
        batch_cfg = load_yaml(batch_path)
        batch_name = str(batch_cfg.get("batch_name") or batch_path.stem)
        status_path = str(args.status_path or batch_cfg.get("status_path") or f"outputs/summary/{batch_name}_status.csv")
        log_dir = str(batch_cfg.get("log_dir") or f"outputs/logs/batch/{batch_name}")
        python_bin = str(args.python_bin or batch_cfg.get("python_bin") or "python")
        experiments = list(batch_cfg.get("experiments") or [])
        for exp in experiments:
            exp.setdefault("batch_config", str(batch_path))
        return batch_name, status_path, log_dir, python_bin, experiments

    configs = args.configs or []
    if not configs:
        raise ValueError("Provide --batch_config or at least one --configs path.")
    batch_name = "ad_hoc_batch"
    status_path = str(args.status_path or "outputs/summary/week7_day2_batch_status.csv")
    log_dir = "outputs/logs/batch/ad_hoc_batch"
    python_bin = str(args.python_bin or "python")
    experiments = [{"config_path": config, "enabled": True} for config in configs]
    return batch_name, status_path, log_dir, python_bin, experiments


def main() -> None:
    args = parse_args()
    batch_name, status_path, log_dir, python_bin, experiments = load_batch(args)
    dry_run = args.dry_run or args.status_only or not args.run

    if args.only_failed:
        failed_names = failed_exp_names(status_path)
        experiments = [
            exp for exp in experiments if str(exp.get("exp_name") or Path(str(exp.get("config_path", ""))).stem) in failed_names
        ]

    enabled_experiments = [exp for exp in experiments if bool(exp.get("enabled", True))]
    if args.max_experiments is not None and args.max_experiments > 0:
        enabled_experiments = enabled_experiments[: args.max_experiments]

    rows: list[dict[str, Any]] = []
    for spec in enabled_experiments:
        if "config_path" not in spec:
            raise ValueError(f"Batch experiment is missing config_path: {spec}")
        row = build_experiment_row(
            spec=spec,
            batch_name=batch_name,
            python_bin=python_bin,
        )
        retry_count = int(spec.get("retry_count", 0) or 0)
        rows.append(
            launch_experiment(
                row=row,
                log_dir=log_dir,
                dry_run=dry_run,
                retry_count=retry_count,
            )
        )

    status_path = write_registry(rows, status_path)
    print(f"Saved batch status to: {status_path}")
    print(f"Batch mode: {'dry-run/status-only' if dry_run else 'run'}")
    print(f"Experiments recorded: {len(rows)}")


if __name__ == "__main__":
    main()

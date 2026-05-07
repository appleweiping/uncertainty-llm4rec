#!/usr/bin/env python3
"""Print server commands for the four-domain Week8 TRUCE pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_DOMAINS = ["beauty", "books", "electronics", "movies"]
DEFAULT_SPLITS = ["valid", "test"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks",
        help="Root containing {domain}_large10000_100neg_{split}_same_candidate directories.",
    )
    parser.add_argument(
        "--output-root",
        default="data/processed/week8_same_candidate",
        help="TRUCE output root for converted processed artifacts.",
    )
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--allow-target-insertion", action="store_true")
    parser.add_argument("--include-ours-adapter-prep", action="store_true")
    parser.add_argument("--emit-json", action="store_true")
    args = parser.parse_args()

    commands = build_commands(
        source_root=args.source_root,
        output_root=args.output_root,
        domains=args.domains,
        splits=args.splits,
        strict_target_in_candidates=not args.allow_target_insertion,
        include_ours_adapter_prep=args.include_ours_adapter_prep,
    )
    if args.emit_json:
        print(json.dumps(commands, indent=2, sort_keys=True))
    else:
        print("# Run from ~/projects/TRUCE-Rec after activating .venv_truce")
        for command in commands:
            print(command)
    return 0


def build_commands(
    *,
    source_root: str,
    output_root: str,
    domains: list[str],
    splits: list[str],
    strict_target_in_candidates: bool = True,
    include_ours_adapter_prep: bool = False,
) -> list[str]:
    commands = []
    for domain in domains:
        for split in splits:
            task_dir = f"{source_root.rstrip('/')}/{domain}_large10000_100neg_{split}_same_candidate"
            out_dir = str(Path(output_root) / f"{domain}_large10000_100neg" / split).replace("\\", "/")
            commands.append(
                "python scripts/convert_week8_same_candidate_to_truce.py "
                f"--task-dir {task_dir} "
                f"--output-dir {out_dir} "
                f"--domain {domain} "
                f"--split {split}"
                + (" --strict-target-in-candidates" if strict_target_in_candidates else "")
            )
        if include_ours_adapter_prep and "test" in splits:
            processed_root = str(Path(output_root) / f"{domain}_large10000_100neg").replace("\\", "/")
            output_dir = f"outputs/server_training/ours_qwen_adapters/{domain}_large10000_100neg"
            commands.append(
                "python scripts/prepare_ours_qwen_adapter_training.py "
                f"--processed-root {processed_root} "
                f"--output-dir {output_dir} "
                f"--domain {domain} "
                "--seed 13"
            )
    return commands


if __name__ == "__main__":
    raise SystemExit(main())

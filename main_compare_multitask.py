from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--exp_names", type=str, nargs="*", default=None)
    parser.add_argument("--finalize_part5", action="store_true")
    parser.add_argument("--part5_input_path", type=str, default=None)
    parser.add_argument("--part5_results_filename", type=str, default="part5_multitask_final_results.csv")
    parser.add_argument("--part5_summary_filename", type=str, default="part5_multitask_final_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_path = REPO_ROOT / "src" / "analysis" / "aggregate_multitask_results.py"
    cmd = [sys.executable, str(script_path), "--output_root", args.output_root]

    if args.domain:
        cmd.extend(["--domain", args.domain])
    if args.model:
        cmd.extend(["--model", args.model])
    if args.exp_names:
        cmd.extend(["--exp_names", *args.exp_names])
    if args.finalize_part5:
        cmd.append("--finalize_part5")
    if args.part5_input_path:
        cmd.extend(["--part5_input_path", args.part5_input_path])
    if args.part5_results_filename:
        cmd.extend(["--part5_results_filename", args.part5_results_filename])
    if args.part5_summary_filename:
        cmd.extend(["--part5_summary_filename", args.part5_summary_filename])

    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()

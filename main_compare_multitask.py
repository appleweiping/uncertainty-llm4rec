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

    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()

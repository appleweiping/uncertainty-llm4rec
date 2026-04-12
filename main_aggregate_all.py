from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def run_step(script_path: Path, output_root: str) -> None:
    cmd = [sys.executable, str(script_path), "--output_root", output_root]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scripts = [
        REPO_ROOT / "src" / "analysis" / "aggregate_domain_results.py",
        REPO_ROOT / "src" / "analysis" / "aggregate_model_results.py",
        REPO_ROOT / "src" / "analysis" / "aggregate_estimator_results.py",
        REPO_ROOT / "src" / "analysis" / "robustness_summary.py",
        REPO_ROOT / "src" / "analysis" / "export_paper_tables.py",
    ]

    for script in scripts:
        run_step(script, args.output_root)

    print("Finished aggregating domain, model, estimator, robustness, and paper-facing summaries.")


if __name__ == "__main__":
    main()

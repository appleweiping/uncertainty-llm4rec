from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.analysis.build_pairwise_coverage_summary import build_pairwise_coverage_summary
from src.analysis.build_part5_figure_pack import build_part5_figure_pack, write_figure_pack_markdown


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--exp_name", type=str, default="beauty_qwen")
    parser.add_argument("--pointwise_exp_name", type=str, default="beauty_qwen_pointwise")
    parser.add_argument("--build_figures", action="store_true")
    parser.add_argument("--build_pairwise_coverage", action="store_true")
    parser.add_argument("--refresh_part5_final", action="store_true")
    return parser.parse_args()


def refresh_part5_final(output_root: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "main_compare_multitask.py"),
        "--output_root",
        output_root,
        "--finalize_part5",
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    args = parse_args()

    if args.refresh_part5_final:
        refresh_part5_final(args.output_root)

    if not args.build_figures and not args.build_pairwise_coverage:
        args.build_figures = True
        args.build_pairwise_coverage = True

    produced: dict[str, Path] = {}
    if args.build_figures:
        produced.update(build_part5_figure_pack(output_root=args.output_root, pointwise_exp_name=args.pointwise_exp_name))
    if args.build_pairwise_coverage:
        produced.update(build_pairwise_coverage_summary(output_root=args.output_root))

    produced["markdown"] = write_figure_pack_markdown(output_root=args.output_root)

    for name, path in produced.items():
        print(f"[Part5 artifacts] {name}: {path}")


if __name__ == "__main__":
    main()

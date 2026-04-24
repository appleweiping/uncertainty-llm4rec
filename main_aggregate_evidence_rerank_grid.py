from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.aggregate_evidence_rerank_grid import save_aggregated_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_summary_path", type=str, required=True, help="Day5 rerank grid summary CSV.")
    parser.add_argument(
        "--output_csv_path",
        type=str,
        default=None,
        help="Aggregated rerank ablation CSV path.",
    )
    parser.add_argument(
        "--output_md_path",
        type=str,
        default=None,
        help="Markdown conclusion path.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="output-repaired",
        help="Output root used when explicit output paths are omitted.",
    )
    return parser.parse_args()


def default_output_paths(grid_summary_path: str | Path, output_root: str) -> tuple[Path, Path]:
    grid_path = Path(grid_summary_path)
    stem = grid_path.stem
    summary_dir = Path(output_root) / "summary"
    return (
        summary_dir / f"{stem}_ablation.csv",
        summary_dir / f"{stem}_conclusion.md",
    )


def main() -> None:
    args = parse_args()
    default_csv_path, default_md_path = default_output_paths(args.grid_summary_path, args.output_root)
    output_csv_path = Path(args.output_csv_path) if args.output_csv_path else default_csv_path
    output_md_path = Path(args.output_md_path) if args.output_md_path else default_md_path

    aggregated_df = save_aggregated_outputs(
        grid_summary_path=args.grid_summary_path,
        output_csv_path=output_csv_path,
        output_md_path=output_md_path,
    )

    print(f"Aggregated {len(aggregated_df)} rerank-grid lambda rows.")
    print(f"Saved ablation CSV to: {output_csv_path}")
    print(f"Saved conclusion markdown to: {output_md_path}")


if __name__ == "__main__":
    main()

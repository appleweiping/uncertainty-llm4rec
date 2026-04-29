"""Run a deeper audit over processed Storyflow observation examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.analysis.dataset_audit import audit_processed_dataset  # noqa: E402


def default_processed_dir(dataset: str, processed_suffix: str) -> Path:
    return ROOT / "data" / "processed" / dataset / processed_suffix


def default_output_dir(dataset: str, processed_suffix: str) -> Path:
    return ROOT / "outputs" / "data_audits" / dataset / processed_suffix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--processed-suffix", required=True)
    parser.add_argument("--processed-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    processed_dir = (
        Path(args.processed_dir)
        if args.processed_dir
        else default_processed_dir(args.dataset, args.processed_suffix)
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir(args.dataset, args.processed_suffix)
    )
    summary = audit_processed_dataset(
        dataset=args.dataset,
        processed_suffix=args.processed_suffix,
        processed_dir=processed_dir,
        output_dir=output_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 1 if summary["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Analyze catalog duplicate titles and optional observation grounding margins."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.analysis import analyze_grounding_diagnostics  # noqa: E402


def _resolve(path: str | Path) -> Path:
    input_path = Path(path)
    return input_path if input_path.is_absolute() else ROOT / input_path


def _processed_dir(dataset: str, processed_suffix: str) -> Path:
    return ROOT / "data" / "processed" / dataset / processed_suffix


def _default_output_dir(dataset: str | None, processed_suffix: str | None) -> Path:
    dataset_name = dataset or "unknown_dataset"
    suffix = processed_suffix or "unknown_processed"
    return ROOT / "outputs" / "grounding_diagnostics" / dataset_name / suffix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--processed-suffix")
    parser.add_argument("--catalog-csv")
    parser.add_argument("--grounded-jsonl")
    parser.add_argument("--manifest-json")
    parser.add_argument("--output-dir")
    parser.add_argument("--margin-threshold", type=float, default=0.03)
    parser.add_argument("--near-miss-threshold", type=float, default=0.70)
    parser.add_argument("--weak-match-threshold", type=float, default=0.40)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--max-groups", type=int, default=50)
    parser.add_argument("--max-cases", type=int, default=50)
    args = parser.parse_args(argv)

    if args.catalog_csv:
        catalog_csv = _resolve(args.catalog_csv)
    elif args.dataset and args.processed_suffix:
        catalog_csv = _processed_dir(args.dataset, args.processed_suffix) / "item_catalog.csv"
    else:
        raise SystemExit("Provide --catalog-csv or both --dataset and --processed-suffix")
    if not catalog_csv.exists():
        raise SystemExit(f"Catalog CSV not found: {catalog_csv}")

    grounded_jsonl = _resolve(args.grounded_jsonl) if args.grounded_jsonl else None
    if grounded_jsonl and not grounded_jsonl.exists():
        raise SystemExit(f"Grounded predictions JSONL not found: {grounded_jsonl}")
    manifest_json = _resolve(args.manifest_json) if args.manifest_json else None
    output_dir = (
        _resolve(args.output_dir)
        if args.output_dir
        else _default_output_dir(args.dataset, args.processed_suffix)
    )
    manifest = analyze_grounding_diagnostics(
        catalog_csv=catalog_csv,
        grounded_jsonl=grounded_jsonl,
        manifest_json=manifest_json if manifest_json and manifest_json.exists() else None,
        output_dir=output_dir,
        dataset=args.dataset,
        processed_suffix=args.processed_suffix,
        margin_threshold=args.margin_threshold,
        near_miss_threshold=args.near_miss_threshold,
        weak_match_threshold=args.weak_match_threshold,
        high_confidence_threshold=args.high_confidence_threshold,
        max_groups=args.max_groups,
        max_cases=args.max_cases,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

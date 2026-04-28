"""Prepare Amazon Reviews 2023 JSONL files when full data is available."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.data import prepare_amazon_from_jsonl, resolve_existing_raw_path  # noqa: E402
from storyflow.utils.config import load_simple_yaml  # noqa: E402


def _config_path(dataset: str) -> Path:
    return ROOT / "configs" / "datasets" / f"{dataset}.yaml"


def _write_missing_report(
    *,
    dataset: str,
    reviews_jsonl: Path,
    metadata_jsonl: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "prepare_readiness_report.md"
    report.write_text(
        "\n".join(
            [
                f"# {dataset} prepare readiness",
                "",
                "当前没有执行 full Amazon preprocessing，因为所需 raw JSONL 文件不存在或未显式提供。",
                "",
                f"- reviews JSONL: `{reviews_jsonl}`",
                f"- metadata JSONL: `{metadata_jsonl}`",
                "- 恢复命令模板:",
                "",
                "```powershell",
                f"python scripts/prepare_amazon_reviews_2023.py --dataset {dataset} --reviews-jsonl {reviews_jsonl} --metadata-jsonl {metadata_jsonl} --output-suffix full",
                "```",
                "",
                "该报告不是 full processed result。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--reviews-jsonl")
    parser.add_argument("--metadata-jsonl")
    parser.add_argument("--output-suffix", default="full")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config_path = _config_path(args.dataset)
    if not config_path.exists():
        raise SystemExit(f"Unknown dataset config: {config_path}")
    config = load_simple_yaml(config_path)
    reviews_jsonl = (
        Path(args.reviews_jsonl)
        if args.reviews_jsonl
        else resolve_existing_raw_path(config, "raw_reviews_path")
    )
    metadata_jsonl = (
        Path(args.metadata_jsonl)
        if args.metadata_jsonl
        else resolve_existing_raw_path(config, "raw_metadata_path")
    )
    output_dir = ROOT / "outputs" / "amazon_reviews_2023" / args.dataset / "prepare"
    if args.dry_run or not reviews_jsonl.exists() or not metadata_jsonl.exists():
        report = _write_missing_report(
            dataset=args.dataset,
            reviews_jsonl=reviews_jsonl,
            metadata_jsonl=metadata_jsonl,
            output_dir=output_dir,
        )
        print(
            json.dumps(
                {
                    "status": "readiness_only",
                    "dataset": args.dataset,
                    "reviews_jsonl_exists": reviews_jsonl.exists(),
                    "metadata_jsonl_exists": metadata_jsonl.exists(),
                    "report": str(report),
                    "full_processed": False,
                    "note": "No full Amazon data was processed.",
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0 if args.dry_run else 2

    summary = prepare_amazon_from_jsonl(
        config=config,
        reviews_jsonl=reviews_jsonl,
        metadata_jsonl=metadata_jsonl,
        output_suffix=args.output_suffix,
        max_records=args.max_records,
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "dataset": summary.dataset,
                "output_dir": str(summary.output_dir),
                "item_count": summary.item_count,
                "interaction_count": summary.interaction_count,
                "user_count": summary.user_count,
                "example_count": summary.example_count,
                "split_counts": summary.split_counts,
                "full_processed": args.max_records is None,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
